# 2026-07-09 Learned Descriptor Replan

## Scope Correction

The paper studies one fixed finite fleet, K=16. The empirical measure is an
order-free quotient of that finite team state, not an infinite-population
approximation and not a requirement that one policy transfer across K.

Cross-K and subpopulation-feedback experiments are therefore diagnostic only.
The formal experiments must compare methods on the same K=16 scenario, training
budget, validation protocol, and final heldout episodes.

The current actor groups all UAV observation rows before producing actions.
Therefore the implemented execution contract is HAP-coordinated centralized
execution, not strict decentralized execution. The paper should either state
that the HAP collects fleet states and dispatches actions, or change the actor
interface before claiming CTDE.

## Evidence from the Last 24 Hours

All values below use 3.5M training steps, three training seeds, and 100 new
heldout episodes per checkpoint unless stated otherwise.

| Method | Descriptor type | Parameters | Heldout reward |
|---|---|---:|---:|
| flat | ordered raw state | 99,983 | -664.965 +/- 9.997 |
| mean | fixed global moment | 80,273 | -580.704 +/- 21.530 |
| hotspot_pool | fixed demand anchors | 85,967 | -532.692 +/- 27.473 |
| anchor_pool | fixed demand + cross anchors | 94,727 | -478.846 +/- 12.990 |
| csd_pool | anchor + global moments | 97,355 | -471.863 +/- 26.627 |
| grid_pool | fixed demand + 3x3 grid | 101,735 | -450.890 +/- 22.913 |

`grid_pool - anchor_pool` has paired hierarchical-bootstrap reward delta
`+27.957`, with 95% CI `[+0.229, +54.746]`. This establishes the value of
spatially distributed local measure statistics. It does not establish a learned
descriptor.

Existing learned descriptors are not a fair final comparison:

- h144 slot-query has about 1.94M actor+critic parameters and its nominal 3.5M
  run stopped at 2.23M; heldout reward was -653.235.
- h64 slot-query completed but heldout reward was -741.911.
- h64/PPO5 DeepSets variants reached roughly -591 to -608 heldout, but they do
  not match the h144/PPO15 main-table protocol.

The supported conclusion is that the current learned implementations are too
large or poorly interfaced, not that descriptor learning is impossible.

## Scope Correction for the Learned Descriptor

Do not make trainable RBF centers, learned grid widths, or a differentiable
version of `grid_pool` the proposed method. Those choices assume that the
handcrafted partition is already the right representation family. The completed
experiments only establish that spatially distributed statistics are useful;
they do not establish that RBF statistics are sufficient or optimal.

Keep `anchor_pool` and `grid_pool` as strong theory-guided baselines. They can
also be used later for an optional hybrid/residual diagnostic, but they must not
define the learned encoder.

## Proposed Architecture

### Lightweight Context-Conditioned Set Encoder

The frozen layer-by-layer implementation and experiment protocol are specified
in `experiments/2026-07-09_learned_set_implementation_spec.md`.

Represent the state as structured entities instead of a flat vector:

- one HAP token containing normalized HAP state, capacity, and load variables;
- a permutation-invariant set of active demand/hotspot tokens;
- a permutation-equivariant set of 16 UAV tokens `(x, y, queue)`;
- learned type embeddings, but no index or ordering embeddings.

Use a deliberately small encoder:

1. Shared token MLPs map each entity type to width 32.
2. One 32-dimensional UAV self-attention block captures pairwise fleet
   interactions. At K=16 its quadratic cost is negligible.
3. Four learned latent queries cross-attend to the UAV and demand tokens. The
   queries are content based: they have no fixed spatial centers and no RBF
   distance term.
4. The HAP actor reads the four pooled latent tokens and the HAP token.
5. A shared UAV decoder reads each contextualized UAV token plus the latent
   tokens, so permuting UAV inputs permutes UAV actions.
6. The critic uses a structurally matched but parameter-separate invariant
   encoder. Actor and critic representation gradients are not shared.

The initial dimensions are `d_token=32`, `M=4`, `heads=4`, and one attention
block. Do not scale every attention module to the PPO hidden width. The target
is below 150k total actor+critic parameters and materially below the existing
1.94M `slot_query` implementation.

This is a generic learned set representation. It includes the successful grid
statistics in its representable function class without fixing the learned
descriptor to a grid.

### Training Objective

Stage L1 trains the encoder end-to-end with the existing role-wise MAPPO
objective only. A descriptor learned by the policy objective is already a
learned descriptor; an auxiliary loss is not a prerequisite.

If L1 is competitive, compare exactly one auxiliary at a time:

- **Action-conditioned self-prediction**: predict the stopped-gradient next
  latent state from the current latent state and the HAP/UAV joint action. The
  predictor is training-only and is never used for rollout or planning.
- **Action-conditioned return prediction**: predict the normalized on-policy
  lambda-return from the actor descriptor and executed joint action. This is a
  model-free, control-aligned alternative to generic state reconstruction.
- **Set reconstruction diagnostic**: reconstruct normalized
  `(x, y, queue)` with an order-free loss. This is an ablation, not the default
  algorithm.

Do not combine all three. The 2024 self-predictive RL results support
stop-gradient latent prediction, while the 2026 Return-Critic result warns that
state reconstruction or state prediction can be misaligned with return. The
comparison must decide which signal is useful in this MEC task.

## Theory Link

Retain the exact fixed-K empirical-measure quotient. Replace the claim that
full-state reconstruction is the defining descriptor criterion with an
approximate control-sufficiency criterion:

- immediate cost/reward consistency for states sharing a descriptor;
- consistency of the next descriptor distribution under the same joint action.

Under standard Lipschitz assumptions, these two errors yield a value-error
bound of approximate MDP-homomorphism/bisimulation form. Full empirical-state
reconstruction is one sufficient way to control these errors, but it is
strictly stronger than necessary. Therefore the decoder and Sinkhorn loss
should not appear in the method name unless the reconstruction ablation wins.

The learned set encoder supplies exact permutation invariance/equivariance by
construction. The control-sufficiency errors, not attention itself, connect the
learned finite-dimensional descriptor to the value bound.

## Experiment Gates

### Stage L1: Learned Descriptor Only

Run three seeds concurrently with the exact main-table protocol:

- K=16, `v7_random_split_hotspots`
- 3.5M steps, h144, PPO15
- no reconstruction, reward prediction, or dynamics prediction
- 100 final heldout episodes at `200000 + 13i`
- actor+critic parameters below 150k
- exact input-permutation equivariance unit test before training

At 1.5M steps:

- continue if the mean validation-best reward is at least -450;
- stop the branch if it is below -500 and all three seeds lag `grid_pool`;
- inspect learned query centers and attention mass before changing losses.

Final promotion requires either:

- better heldout reward than `grid_pool` with a paired interval excluding zero;
  or
- reward within 10 points of `grid_pool` with lower variance or a materially
  better representation-fidelity diagnostic.

### Stage L2: One Auxiliary at a Time

Only enter L2 if L1 is within 20-30 reward points of `grid_pool`.

Compare one auxiliary at a time on seed1:

1. stop-gradient action-conditioned one-step latent prediction;
2. action-conditioned lambda-return prediction;
3. full empirical-state set reconstruction on normalized `(x, y, queue)`.

Select at most one objective for final three-seed confirmation. Monitor
descriptor effective rank, auxiliary gradient norm relative to the PPO actor
gradient, and heldout control reward. An auxiliary that improves its own loss
but degrades return is rejected.

### Stage L2b: Optional Hybrid Diagnostic

Only if the learned-only method is competitive but does not beat `grid_pool`,
concatenate a stop-gradient `grid_pool` descriptor with the learned descriptor.
This tests whether the learned encoder discovers residual information beyond
the handcrafted spatial statistics. It is not the primary method and must not
replace the learned-only comparison.

### Stage L3: Paper Confirmation

For the selected architecture:

- three full-budget seeds;
- 100-episode fixed-K heldout comparison;
- parameter count and inference latency;
- input-permutation equivariance test;
- learned-query visualization;
- reconstruction or latent-prediction diagnostic only if L2 selected it.

## Paper Decision

If the learned set descriptor succeeds, the algorithm contribution is a
low-complexity, control-sufficient learned set representation for fixed finite
major-minor cooperative control. The exact finite-K quotient, exact
invariance/equivariance, and approximate control-sufficiency error analysis
provide the theoretical frame.

If it fails, do not force a SetRec claim. Use `grid_pool` as a theory-guided
spatial quantization method, remove learned-reconstruction claims from the title,
and position the work as finite-fleet control rather than representation
learning.

## Relevant References

- Bayraktar, Baeuerle, Kara, *Finite Approximations for Mean-Field Type
  Multi-agent Control and Their Near Optimality*, Applied Mathematics &
  Optimization, 2025.
- Panangaden et al., *Policy Gradient Methods in the Presence of Symmetries and
  State Abstractions*, JMLR, 2024.
- Hao et al., *Boosting Multiagent Reinforcement Learning via Permutation
  Invariant and Permutation Equivariant Networks*, ICLR 2023.
- Park, Seong, Ko, *SPECTra: Scalable Multi-Agent Reinforcement Learning with
  Permutation-Free Networks*, IEEE Access, accepted 2026.
- Lee et al., *Set Transformer: A Framework for Attention-based
  Permutation-Invariant Neural Networks*, ICML 2019.
- Kortvelesy, Morad, Prorok, *Permutation-Invariant Set Autoencoders with
  Fixed-Size Embeddings for Multi-Agent Learning*, AAMAS 2023.
- Guan et al., *Efficient Multi-agent Communication via Self-supervised
  Information Aggregation*, NeurIPS 2022.
- Ni et al., *Bridging State and History Representations: Understanding
  Self-Predictive RL*, ICLR 2024.
- Lu et al., *Return-Critic: Bridging Goal Discrepancy for Efficient Visual
  Reinforcement Learning*, ICML 2026.
- Liang et al., *Reconstruction-Guided Policy: Enhancing Decision-Making through
  Agent-Wise State Consistency*, ICLR 2025.
- Kang et al., *MA2E: Addressing Partial Observability in Multi-Agent
  Reinforcement Learning with Masked Auto-Encoder*, ICLR 2025.
