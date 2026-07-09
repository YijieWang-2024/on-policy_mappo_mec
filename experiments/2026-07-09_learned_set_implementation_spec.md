# 2026-07-09 Learned Set Implementation Specification

## Decision

The first learned descriptor must be a small generic entity-set encoder, not a
trainable RBF/grid model. `grid_pool` remains the strongest handcrafted
baseline.

The first formal learned run keeps the established `hidden_size=144` and
`ppo_epoch=15` so that the downstream policy capacity and PPO update budget
match the existing v7 table. The set encoder has its own fixed dimensions and
must not use `args.hidden_size` or `args.layer_N`.

## Input Contract

Split `public` into:

- HAP/global token: `public[:3]` plus the six resource-context values, 9 dims;
- four hotspot tokens: `public[3:23].reshape(4, 5)`;
- sixteen UAV tokens: normalized `(x, y, queue)`, 3 dims.

Inactive hotspot tokens use a key-padding mask based on hotspot weight. There
are no UAV indices, hotspot indices, positional embeddings, fixed anchors, or
RBF features.

## Encoder

Actor and critic own separate instances of the following encoder.

### Entity Embeddings

Each entity type has an explicit two-linear MLP:

```text
UAV:     Linear(3, 32) -> ReLU -> Linear(32, 32)
hotspot: Linear(5, 32) -> ReLU -> Linear(32, 32)
HAP:     Linear(9, 32) -> ReLU -> Linear(32, 32)
```

Do not LayerNorm the raw 3/5/9-dimensional physical features. They are already
normalized by the environment, and per-token raw LayerNorm would mix physical
coordinates and queue/load semantics.

### Context Block

Concatenate one HAP token, four hotspot tokens, and sixteen UAV tokens. Apply
one pre-norm Transformer block:

```text
x = x + MHA(LayerNorm(x), LayerNorm(x), LayerNorm(x))
x = x + Linear(64, 32)(ReLU(Linear(32, 64)(LayerNorm(x))))
```

Use `d_model=32`, `heads=4`, attention dropout 0, residual dropout 0, and one
block only. Mask inactive hotspot keys.

### Latent Queries

Use four learned query rows `Q in R^(4 x 32)`. Initialize the rows
orthogonally, not with `Normal(0, 0.02)`. Before the first cross-attention,
LayerNorm both the queries and contextualized UAV tokens:

```text
q = q + MHA(LayerNorm(q), LayerNorm(uav_tokens), LayerNorm(uav_tokens))
q = q + Linear(64, 32)(ReLU(Linear(32, 64)(LayerNorm(q))))
```

The query rows have no coordinates or spatial interpretation. Orthogonal
initialization provides distinct directions; pre-norm makes their absolute
initial scale unimportant.

### Descriptor

Softmax attention alone can obscure population mass. Preserve a DeepSets path:

```text
uav_mean = mean(contextualized_uav_tokens, dim=UAV)
descriptor_input = concat(uav_mean, flatten(q))  # 32 + 4*32 = 160
descriptor = LayerNorm(ReLU(Linear(160, 64)(descriptor_input)))
```

This gives a 64-dimensional invariant descriptor. The mean branch is required;
it is not an optional ablation in the first implementation.

## Actor Readout

Use explicit two-hidden-layer readouts:

```text
HAP: concat(contextualized_hap[32], descriptor[64])
     -> Linear(96, 144) -> ReLU -> LayerNorm
     -> Linear(144, 144) -> ReLU -> LayerNorm

UAV i: concat(contextualized_uav_i[32],
              contextualized_hap[32],
              descriptor[64])
       -> Linear(128, 144) -> ReLU -> LayerNorm
       -> Linear(144, 144) -> ReLU -> LayerNorm
```

The UAV readout is shared across all UAVs. Do not add another per-UAV
cross-attention block; the contextual UAV token already carries entity
interactions.

Keep the existing action heads and their initialization:

- velocity mean heads: orthogonal gain `0.01`;
- Gaussian `logstd=-1.9`;
- Beta head weights zero, initialized to `alpha=beta=2`.

## Critic

Use an independently parameterized encoder with the same 32/4/64 dimensions.
Its invariant readout is:

```text
concat(contextualized_hap[32], descriptor[64])
-> Linear(96, 144) -> ReLU -> LayerNorm
-> Linear(144, 144) -> ReLU -> LayerNorm
-> Linear(144, 1)
```

Do not share actor/critic encoder parameters and do not use `shared_grad`.

Estimated actor+critic parameter count is about 175k with width 144. A later
width-96 efficiency ablation is about 124k. The implementation must report the
actual count.

## Initialization

- entity/readout/fusion linear layers: orthogonal weights;
- hidden ReLU layers: gain `sqrt(2)`;
- biases: zero;
- `nn.MultiheadAttention`: PyTorch Xavier defaults;
- four query rows: orthogonal rows;
- LayerNorm scale one and bias zero;
- no dropout and no weight decay in the first experiment.

## Matched PPO Configuration

Use the current v7 main-table protocol:

```text
K=16
n_rollout_threads=8
episode_length=200
num_mini_batch=1
ppo_epoch=15
hidden_size=144
lr=5e-4
critic_lr=5e-4
clip_param=0.2
entropy_coef=0.01
gamma=0.99
gae_lambda=0.95
ValueNorm=true
max_grad_norm=10
target_kl=None
```

`layer_N` is irrelevant inside the new encoder and readouts. Retain it in the
global config only for compatibility.

Do not reduce PPO epochs or hidden width before establishing the matched
result. Existing grid runs have median KL near 0.005; PPO15 is not currently
demonstrated to be excessive. Test PPO5 only if the learned model crosses the
optimization failure thresholds below.

## Required Tests Before Training

1. UAV permutation changes UAV actions by the same permutation and leaves the
   HAP action/value unchanged.
2. Hotspot permutation leaves all outputs unchanged.
3. Inactive hotspot padding leaves outputs unchanged.
4. All four query rows receive finite nonzero gradients.
5. Actor and critic encoder parameters are disjoint.
6. One grouped PPO update runs without NaN/Inf.
7. Report actor, critic, and total parameter counts.

## Experiment Order

### Batch 0: Implementation and 20k Smoke

Run one seed for 20k steps. This is only a numerical test, not a reward gate.
Log KL, clip fraction, actor/critic gradient norm, query-pair cosine,
normalized attention entropy, and descriptor effective rank.

### Batch 1: 1.5M Interface/Role Gate

Run these three seed-1 jobs concurrently:

| Job | Descriptor | Role-wise PPO | Purpose |
|---|---|---:|---|
| L0-matched | learned set | off | exact match to current v7 table |
| L0-role | learned set | on | test HAP/UAV gradient balance |
| G-role | grid_pool | on | control for the role-wise change |

All other settings remain h144/PPO15.

After the first 200k post-warmup updates, an optimization branch fails if any
of the following persists:

- NaN/Inf;
- `p95(approx_kl) > 0.02`;
- `p95(clip_fraction) > 0.15`;
- `p95(actor_grad_norm) > 2`;
- all query-pair cosine similarities exceed 0.98;
- descriptor effective rank is below 8.

At 1.5M:

- continue a learned branch if validation-best reward is at least `-500`;
- stop it if validation-best is below `-550` and the last five evaluations
  show no improvement;
- in the gray interval `[-550, -500)`, continue only the better learned
  branch.

The role-wise setting is selected only by the paired interpretation of
`L0-role - L0-matched` and `G-role - existing grid`. If role-wise improves both,
the final grid baseline must also be rerun role-wise.

### Batch 2: Formal Learned-Only Confirmation

Continue the selected seed-1 job to 3.5M and launch seeds 2 and 3 with the
identical frozen configuration. Evaluate the best checkpoint of each seed on
the same 100 heldout episode seeds used by the existing table.

Do not select h96 or PPO5 from short-run reward. They are conditional
ablations:

- run `hidden/readout width=96` only after h144 proves the architecture viable;
- run PPO5 only if PPO15 violates the KL/clip thresholds.

### Batch 3: One Auxiliary at a Time

Enter only if learned-only finishes within 20-30 heldout reward points of
`grid_pool`.

Use one seed per objective:

1. stop-gradient action-conditioned next-latent prediction, coefficient 0.05;
2. action-conditioned normalized lambda-return prediction, Huber loss,
   coefficient 0.05;
3. normalized `(x,y,queue)` set reconstruction, coefficient 0.01.

Ramp the auxiliary coefficient linearly from zero during the first 100k steps.
Reject an auxiliary if its own loss improves while heldout control or PPO
optimization diagnostics worsen. Promote at most one auxiliary to three seeds.

## Reporting

The final table must report:

- heldout reward and physical cost components;
- actor, critic, and total parameters;
- single-step inference latency;
- permutation error;
- query cosine/attention entropy/effective rank;
- PPO KL and clip fraction;
- auxiliary diagnostic only for the selected auxiliary.
