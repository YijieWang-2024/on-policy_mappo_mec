"""MEC-specific actor and critic networks.

These modules implement the major/minor action contract and convert canonical
MEC obs/share_obs into mean or flat population representations. The policy
wrapper in ``mec_policy.py`` owns optimizers and exposes the MAPPO policy API.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Beta, Normal

from onpolicy.algorithms.utils.mlp import MLPBase, MLPLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.util import check, init
from onpolicy.envs.mec.observation import (
    AGENT_OBS_DIM,
    HOTSPOT_FEATURE_DIM,
    MAX_HOTSPOTS,
    OWN_SLICE,
    PHYSICAL_PUBLIC_STATE_DIM,
    PUBLIC_SLICE,
    PUBLIC_STATE_DIM,
    UAV_STATE_DIM,
    team_state_dim,
)
from onpolicy.utils.util import get_shape_from_obs_space

_EPS = 1e-6
_BETA_EPS = 1e-4
_ANCHOR_STAT_DIM = 4
_FIXED_ANCHORS = torch.tensor(
    [
        [0.50, 0.50],
        [0.25, 0.50],
        [0.75, 0.50],
        [0.50, 0.25],
        [0.50, 0.75],
    ],
    dtype=torch.float32,
)
_GRID_ANCHORS = torch.tensor(
    [(x, y) for x in (0.2, 0.5, 0.8) for y in (0.2, 0.5, 0.8)],
    dtype=torch.float32,
)
_ANCHOR_SIGMA_NORM = 0.10


def _mlp(args, input_dim):
    return MLPBase(args, (int(input_dim),))


def _softplus_inverse(x: float) -> float:
    if x <= 0.0:
        raise ValueError("softplus inverse input must be positive")
    return math.log(math.expm1(x))


def _init_beta_head(layer, target_alpha: float, target_beta: float):
    nn.init.constant_(layer.weight, 0.0)
    nn.init.constant_(layer.bias[0], _softplus_inverse(target_alpha - _BETA_EPS))
    nn.init.constant_(layer.bias[1], _softplus_inverse(target_beta - _BETA_EPS))
    return layer


def _pool_dim(arch: str) -> int:
    if arch == "hotspot_pool":
        return MAX_HOTSPOTS * _ANCHOR_STAT_DIM
    if arch == "anchor_pool":
        return (MAX_HOTSPOTS + len(_FIXED_ANCHORS)) * _ANCHOR_STAT_DIM
    if arch == "grid_pool":
        return (MAX_HOTSPOTS + len(_GRID_ANCHORS)) * _ANCHOR_STAT_DIM
    if arch == "csd_pool":
        return _pool_dim("anchor_pool") + 2 * UAV_STATE_DIM
    raise ValueError(f"not a pooled MEC arch: {arch}")


def _anchor_pool(public: torch.Tensor, uavs: torch.Tensor, arch: str) -> torch.Tensor:
    physical = public[:, :PHYSICAL_PUBLIC_STATE_DIM]
    hotspot_tokens = physical[:, 3:].reshape(-1, MAX_HOTSPOTS, HOTSPOT_FEATURE_DIM)
    anchors = hotspot_tokens[:, :, :2]
    active = (hotspot_tokens[:, :, 4:5] > 0.0).to(uavs.dtype)
    if arch in {"anchor_pool", "grid_pool"}:
        fixed_anchors = _FIXED_ANCHORS if arch == "anchor_pool" else _GRID_ANCHORS
        fixed = fixed_anchors.to(device=uavs.device, dtype=uavs.dtype).expand(uavs.shape[0], -1, -1)
        anchors = torch.cat([anchors, fixed], dim=1)
        active = torch.cat([active, torch.ones_like(fixed[:, :, :1])], dim=1)

    xy = uavs[:, :, :2]
    queue = uavs[:, :, 2:3]
    diff = xy[:, :, None, :] - anchors[:, None, :, :]
    radial = torch.exp(-torch.sum(diff * diff, dim=-1) / (2.0 * _ANCHOR_SIGMA_NORM**2))
    radial = radial * active.transpose(1, 2)
    denom = radial.sum(dim=1).clamp_min(_EPS)
    xy_offset_mean = (radial[..., None] * diff).sum(dim=1) / denom[..., None]
    queue_mean = (radial[..., None] * queue[:, :, None, :]).sum(dim=1) / denom[..., None]
    mass = (radial.sum(dim=1, keepdim=False) / max(float(uavs.shape[1]), 1.0))[..., None]
    stats = torch.cat([xy_offset_mean, queue_mean, mass], dim=-1) * active
    return stats.reshape(uavs.shape[0], -1)


def _pooled_rep(public: torch.Tensor, uavs: torch.Tensor, arch: str) -> torch.Tensor:
    if arch == "csd_pool":
        fleet_stats = torch.cat([uavs.mean(dim=1), uavs.std(dim=1, unbiased=False)], dim=-1)
        return torch.cat([_anchor_pool(public, uavs, "anchor_pool"), fleet_stats], dim=-1)
    return _anchor_pool(public, uavs, arch)


class _SplitMLPBase(nn.Module):
    def __init__(self, args, part_dims):
        super().__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self.norms = nn.ModuleList(nn.LayerNorm(int(dim)) for dim in part_dims)
        self.mlp = MLPLayer(
            sum(int(dim) for dim in part_dims),
            args.hidden_size,
            args.layer_N,
            args.use_orthogonal,
            args.use_ReLU,
        )

    def forward(self, x):
        raise RuntimeError("_SplitMLPBase requires forward_parts")

    def forward_parts(self, parts):
        if self._use_feature_normalization:
            parts = [norm(part) for norm, part in zip(self.norms, parts)]
        return self.mlp(torch.cat(parts, dim=-1))


class _TokenMLP(nn.Module):
    def __init__(self, args, input_dim: int):
        super().__init__()
        self.mlp = MLPLayer(
            int(input_dim),
            args.hidden_size,
            args.layer_N,
            args.use_orthogonal,
            args.use_ReLU,
        )

    def forward(self, x):
        return self.mlp(x)


def _deepsets_token_dim(feature_mode: str) -> int:
    if feature_mode == "own":
        return UAV_STATE_DIM
    if feature_mode == "geo":
        return UAV_STATE_DIM + 3 + MAX_HOTSPOTS * 4
    if feature_mode == "ref":
        return UAV_STATE_DIM + 3 + (MAX_HOTSPOTS + len(_FIXED_ANCHORS)) * 5
    raise ValueError("mec_deepsets_features must be own|geo|ref")


def _deepsets_tokens(public: torch.Tensor, uavs: torch.Tensor, feature_mode: str) -> torch.Tensor:
    if feature_mode == "own":
        return uavs

    physical = public[:, :PHYSICAL_PUBLIC_STATE_DIM]
    hap_xy = physical[:, :2]
    hap_diff = uavs[:, :, :2] - hap_xy[:, None, :]
    hap_rel = torch.cat([hap_diff, torch.sum(hap_diff * hap_diff, dim=-1, keepdim=True)], dim=-1)
    hotspot_tokens = physical[:, 3:].reshape(-1, MAX_HOTSPOTS, HOTSPOT_FEATURE_DIM)
    hot_xy = hotspot_tokens[:, :, :2]
    hot_weight = hotspot_tokens[:, :, 4:5].to(uavs.dtype)
    hot_diff = uavs[:, :, None, :2] - hot_xy[:, None, :, :]
    hot_r2 = torch.sum(hot_diff * hot_diff, dim=-1, keepdim=True)
    if feature_mode == "geo":
        hot_rel = torch.cat(
            [hot_diff, hot_r2, hot_weight[:, None, :, :].expand(-1, uavs.shape[1], -1, -1)],
            dim=-1,
        )
        return torch.cat([uavs, hap_rel, hot_rel.reshape(uavs.shape[0], uavs.shape[1], -1)], dim=-1)

    fixed_xy = _FIXED_ANCHORS.to(device=uavs.device, dtype=uavs.dtype).expand(uavs.shape[0], -1, -1)
    refs = torch.cat([hot_xy, fixed_xy], dim=1)
    weight = torch.cat([hot_weight, torch.zeros_like(fixed_xy[:, :, :1])], dim=1)
    is_fixed = torch.cat([torch.zeros_like(hot_weight), torch.ones_like(fixed_xy[:, :, :1])], dim=1)
    diff = uavs[:, :, None, :2] - refs[:, None, :, :]
    r2 = torch.sum(diff * diff, dim=-1, keepdim=True)
    rel = torch.cat(
        [
            diff,
            r2,
            weight[:, None, :, :].expand(-1, uavs.shape[1], -1, -1),
            is_fixed[:, None, :, :].expand(-1, uavs.shape[1], -1, -1),
        ],
        dim=-1,
    )
    return torch.cat([uavs, hap_rel, rel.reshape(uavs.shape[0], uavs.shape[1], -1)], dim=-1)


class _MultiHeadDeepSets(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_mode = str(getattr(args, "mec_deepsets_features", "geo"))
        token_dim = _deepsets_token_dim(self.feature_mode)
        heads = int(getattr(args, "mec_deepsets_heads", 4))
        if heads < 1:
            raise ValueError("mec_deepsets_heads must be >= 1")
        self.pool = str(getattr(args, "mec_deepsets_pool", "mean_std"))
        if self.pool not in {"mean", "mean_std"}:
            raise ValueError("mec_deepsets_pool must be mean|mean_std")
        self.heads = nn.ModuleList(_TokenMLP(args, token_dim) for _ in range(heads))
        pool_mult = 2 if self.pool == "mean_std" else 1
        self.fuse = _mlp(args, heads * pool_mult * int(args.hidden_size))
        self.output_dim = int(args.hidden_size)

    def forward(self, public: torch.Tensor, uavs: torch.Tensor) -> torch.Tensor:
        tokens = _deepsets_tokens(public, uavs, self.feature_mode)
        pooled = []
        flat = tokens.reshape(-1, tokens.shape[-1])
        for head in self.heads:
            h = head(flat).reshape(tokens.shape[0], tokens.shape[1], -1)
            stats = [h.mean(dim=1)]
            if self.pool == "mean_std":
                stats.append(h.std(dim=1, unbiased=False))
            pooled.append(torch.cat(stats, dim=-1))
        return self.fuse(torch.cat(pooled, dim=-1))


class _AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"attention dim {dim} must divide heads {heads}")
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(query, memory, memory, need_weights=False)
        x = self.norm1(query + attended)
        return self.norm2(x + self.ffn(x))


class _SlotEncoder(nn.Module):
    def __init__(self, args, atom_dim: int):
        super().__init__()
        self.dim = int(args.hidden_size)
        self.slots = int(getattr(args, "mec_slot_num", 4))
        heads = int(getattr(args, "mec_slot_heads", 4))
        blocks = int(getattr(args, "mec_slot_blocks", 1))
        self.atom = _mlp(args, atom_dim)
        self.blocks = nn.ModuleList(_AttentionBlock(self.dim, heads) for _ in range(blocks))
        self.seeds = nn.Parameter(torch.empty(1, self.slots, self.dim))
        nn.init.normal_(self.seeds, std=0.02)
        self.pool = _AttentionBlock(self.dim, heads)

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        tokens = self.atom(atoms)
        for block in self.blocks:
            tokens = block(tokens, tokens)
        seeds = self.seeds.expand(atoms.shape[0], -1, -1)
        return self.pool(seeds, tokens)


class _SlotReadout(nn.Module):
    def __init__(self, args, query_dim: int):
        super().__init__()
        self.query = _mlp(args, query_dim)
        self.attn = _AttentionBlock(int(args.hidden_size), int(getattr(args, "mec_slot_heads", 4)))

    def forward(self, query_input: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        query = self.query(query_input)
        context = self.attn(query, slots)
        return torch.cat([query, context], dim=-1)


class _SlotDecoder(nn.Module):
    def __init__(self, args, num_uavs: int):
        super().__init__()
        hidden = int(args.hidden_size)
        self.num_uavs = int(num_uavs)
        self.net = nn.Sequential(
            nn.Linear(int(getattr(args, "mec_slot_num", 4)) * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.num_uavs * UAV_STATE_DIM),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return self.net(slots.reshape(slots.shape[0], -1)).reshape(
            -1, self.num_uavs, UAV_STATE_DIM
        )


def _sinkhorn_set_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cost = torch.cdist(pred, target).pow(2)
    log_p = -cost / 0.1
    for _ in range(20):
        log_p = log_p - torch.logsumexp(log_p, dim=2, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
    return (log_p.exp() * cost).sum(dim=(1, 2)).div(pred.shape[1]).mean()


class MECActor(nn.Module):
    VEL_DIM = 2
    ACT_DIM = 3

    def __init__(self, args, obs_space, action_space, num_agents, device):
        super().__init__()
        self.num_agents = int(num_agents)
        self.num_uavs = self.num_agents - 1
        self.obs_dim = int(get_shape_from_obs_space(obs_space)[0])
        self.arch = getattr(args, "mec_policy_arch", "mean")
        self.hidden_size = args.hidden_size
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_rolewise_loss = bool(getattr(args, "mec_rolewise_loss", False))
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.obs_dim != AGENT_OBS_DIM:
            raise ValueError(f"MEC obs dim must be {AGENT_OBS_DIM}, got {self.obs_dim}")
        if args.use_recurrent_policy or args.use_naive_recurrent_policy:
            raise NotImplementedError("MEC mean/flat policy is feed-forward only")
        if self.arch not in {"mean", "flat", "hotspot_pool", "anchor_pool", "grid_pool", "csd_pool", "slot_query", "mhd_deepsets"}:
            raise ValueError("MEC supports --mec_policy_arch mean|flat|hotspot_pool|anchor_pool|grid_pool|csd_pool|slot_query|mhd_deepsets")

        if self.arch == "mean":
            major_dim = PUBLIC_STATE_DIM + UAV_STATE_DIM
            minor_dim = UAV_STATE_DIM + PUBLIC_STATE_DIM + UAV_STATE_DIM
        elif self.arch == "flat":
            flat_dim = PUBLIC_STATE_DIM + self.num_uavs * UAV_STATE_DIM
            major_dim = flat_dim
            minor_dim = UAV_STATE_DIM + flat_dim
        elif self.arch == "slot_query":
            self.minor_encoder = _SlotEncoder(args, UAV_STATE_DIM)
            self.major_encoder = (
                self.minor_encoder
                if bool(getattr(args, "mec_slot_actor_share_encoder", False))
                else _SlotEncoder(args, UAV_STATE_DIM)
            )
            self.minor_readout = _SlotReadout(args, UAV_STATE_DIM + PUBLIC_STATE_DIM)
            self.major_readout = _SlotReadout(args, PUBLIC_STATE_DIM)
            self.reconstruction_decoder = _SlotDecoder(args, self.num_uavs)
            major_dim = minor_dim = 2 * self.hidden_size
        elif self.arch == "mhd_deepsets":
            self.descriptor = _MultiHeadDeepSets(args)
            major_dim = minor_dim = None
        else:
            pool_dim = _pool_dim(self.arch)
            major_dim = PUBLIC_STATE_DIM + pool_dim
            minor_dim = UAV_STATE_DIM + PUBLIC_STATE_DIM + pool_dim

        if self.arch == "mhd_deepsets":
            self.major_base = _SplitMLPBase(args, [PUBLIC_STATE_DIM, self.descriptor.output_dim])
            self.minor_base = _SplitMLPBase(
                args, [UAV_STATE_DIM, PUBLIC_STATE_DIM, self.descriptor.output_dim]
            )
        else:
            self.major_base = _mlp(args, major_dim)
            self.minor_base = _mlp(args, minor_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]

        def init_(module, gain=0.01):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        logstd_init = float(getattr(args, "mec_logstd_init", -1.9))
        self.major_mean = init_(nn.Linear(self.hidden_size, self.VEL_DIM))
        self.major_logstd = nn.Parameter(torch.full((self.VEL_DIM,), logstd_init))
        self.minor_mean = init_(nn.Linear(self.hidden_size, self.VEL_DIM))
        self.minor_logstd = nn.Parameter(torch.full((self.VEL_DIM,), logstd_init))

        beta_alpha_init = float(getattr(args, "mec_beta_alpha_init", 2.0))
        beta_eta_init = float(getattr(args, "mec_beta_eta_init", 2.0))
        self.minor_beta = _init_beta_head(
            nn.Linear(self.hidden_size, 2),
            target_alpha=beta_alpha_init,
            target_beta=beta_eta_init,
        )
        self.to(device)

    def _team(self, obs):
        obs = check(obs).to(**self.tpdv)
        if obs.shape[-1] != self.obs_dim or obs.shape[0] % self.num_agents != 0:
            raise ValueError("MEC actor requires complete K+1 canonical obs rows")
        return obs.reshape(-1, self.num_agents, self.obs_dim)

    def _features(self, obs):
        team = self._team(obs)
        public = team[:, 0, PUBLIC_SLICE]
        uavs = team[:, 1:, OWN_SLICE]
        if self.arch == "mean":
            rep = uavs.mean(dim=1)
        elif self.arch == "flat":
            rep = uavs.reshape(uavs.shape[0], -1)
        elif self.arch == "slot_query":
            return self._slot_features(public, uavs)
        elif self.arch == "mhd_deepsets":
            return self._deepsets_features(public, uavs)
        else:
            rep = _pooled_rep(public, uavs, self.arch)

        major_in = torch.cat([public, rep], dim=-1)
        minor_in = torch.cat(
            [
                uavs,
                public[:, None, :].expand(-1, self.num_uavs, -1),
                rep[:, None, :].expand(-1, self.num_uavs, -1),
            ],
            dim=-1,
        )
        major_feat = self.major_base(major_in).unsqueeze(1)
        minor_feat = self.minor_base(minor_in.reshape(-1, minor_in.shape[-1]))
        minor_feat = minor_feat.reshape(-1, self.num_uavs, self.hidden_size)
        return torch.cat([major_feat, minor_feat], dim=1).reshape(-1, self.hidden_size)

    def _deepsets_features(self, public, uavs):
        rep = self.descriptor(public, uavs)
        public_uav = public[:, None, :].expand(-1, self.num_uavs, -1)
        rep_uav = rep[:, None, :].expand(-1, self.num_uavs, -1)
        major_feat = self.major_base.forward_parts([public, rep]).unsqueeze(1)
        minor_feat = self.minor_base.forward_parts(
            [
                uavs.reshape(-1, UAV_STATE_DIM),
                public_uav.reshape(-1, PUBLIC_STATE_DIM),
                rep_uav.reshape(-1, self.descriptor.output_dim),
            ]
        )
        minor_feat = minor_feat.reshape(-1, self.num_uavs, self.hidden_size)
        return torch.cat([major_feat, minor_feat], dim=1).reshape(-1, self.hidden_size)

    def _slot_features(self, public, uavs):
        public_uav = public[:, None, :].expand(-1, self.num_uavs, -1)
        minor_slots = self.minor_encoder(uavs)
        major_slots = self.major_encoder(uavs)
        minor_in = torch.cat([uavs, public_uav], dim=-1)
        minor_feat = self.minor_base(self.minor_readout(minor_in, minor_slots).reshape(-1, 2 * self.hidden_size))
        minor_feat = minor_feat.reshape(-1, self.num_uavs, self.hidden_size)
        major_feat = self.major_base(self.major_readout(public[:, None, :], major_slots).squeeze(1)).unsqueeze(1)
        return torch.cat([major_feat, minor_feat], dim=1).reshape(-1, self.hidden_size)

    def reconstruction_loss(self, cent_obs):
        if self.arch != "slot_query":
            raise RuntimeError("reconstruction loss requires slot_query")
        cent_obs = check(cent_obs).to(**self.tpdv)
        uavs = cent_obs[:, PUBLIC_STATE_DIM:].reshape(-1, self.num_uavs, UAV_STATE_DIM)
        slots = self.minor_encoder(uavs)
        pred = self.reconstruction_decoder(slots)
        return _sinkhorn_set_loss(pred, uavs)

    def _dists(self, features):
        major_mean = self.major_mean(features)
        major_std = torch.exp(self.major_logstd).expand_as(major_mean)
        minor_mean = self.minor_mean(features)
        minor_std = torch.exp(self.minor_logstd).expand_as(minor_mean)
        ab = torch.nn.functional.softplus(self.minor_beta(features)) + _BETA_EPS
        return (
            Normal(major_mean, major_std),
            Normal(minor_mean, minor_std),
            Beta(ab[:, 0:1], ab[:, 1:2]),
        )

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        rnn_states = check(rnn_states).to(**self.tpdv)
        is_major = (check(obs).to(**self.tpdv)[:, 0:1] > 0.5).float()

        features = self._features(obs)
        major, minor_v, minor_b = self._dists(features)

        v_major = major.mean if deterministic else major.rsample()
        v_minor = minor_v.mean if deterministic else minor_v.rsample()
        b_minor = minor_b.mean if deterministic else minor_b.rsample()

        vel = is_major * v_major + (1.0 - is_major) * v_minor
        beta = (1.0 - is_major) * b_minor
        actions = torch.cat([vel, beta], dim=-1)

        lp_major = major.log_prob(v_major).sum(-1, keepdim=True)
        lp_minor = minor_v.log_prob(v_minor).sum(-1, keepdim=True) + minor_b.log_prob(
            b_minor.clamp(_EPS, 1.0 - _EPS)
        )
        return actions, is_major * lp_major + (1.0 - is_major) * lp_minor, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        obs_t = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        features = self._features(obs_t)
        is_major = (obs_t[:, 0:1] > 0.5).float()
        major, minor_v, minor_b = self._dists(features)

        vel = action[:, : self.VEL_DIM]
        beta = action[:, self.VEL_DIM : self.VEL_DIM + 1].clamp(_EPS, 1.0 - _EPS)
        lp_major = major.log_prob(vel).sum(-1, keepdim=True)
        lp_minor = minor_v.log_prob(vel).sum(-1, keepdim=True) + minor_b.log_prob(beta)
        ent_major = major.entropy().sum(-1, keepdim=True)
        ent_minor = minor_v.entropy().sum(-1, keepdim=True) + minor_b.entropy()
        action_log_probs = is_major * lp_major + (1.0 - is_major) * lp_minor
        entropy = is_major * ent_major + (1.0 - is_major) * ent_minor

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        if self._use_rolewise_loss:
            weights = active_masks if self._use_policy_active_masks and active_masks is not None else torch.ones_like(entropy)
            major_w = weights * is_major
            minor_w = weights * (1.0 - is_major)
            dist_entropy = 0.5 * (
                (entropy * major_w).sum() / major_w.sum().clamp_min(1.0)
                + (entropy * minor_w).sum() / minor_w.sum().clamp_min(1.0)
            )
        elif self._use_policy_active_masks and active_masks is not None:
            dist_entropy = (entropy * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = entropy.mean()
        return action_log_probs, dist_entropy


class MECCritic(nn.Module):
    def __init__(self, args, cent_obs_space, num_agents, device, actor_encoder=None):
        super().__init__()
        self.num_agents = int(num_agents)
        self.num_uavs = self.num_agents - 1
        self.state_dim = int(get_shape_from_obs_space(cent_obs_space)[0])
        self.arch = getattr(args, "mec_critic_arch", None) or getattr(args, "mec_policy_arch", "mean")
        self.hidden_size = args.hidden_size
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)

        if self.state_dim != team_state_dim(self.num_agents):
            raise ValueError("MEC critic requires canonical team state")
        if self.arch == "mean":
            rep_dim = UAV_STATE_DIM
        elif self.arch == "flat":
            rep_dim = self.num_uavs * UAV_STATE_DIM
        elif self.arch in {"hotspot_pool", "anchor_pool", "grid_pool", "csd_pool"}:
            rep_dim = _pool_dim(self.arch)
        elif self.arch == "mhd_deepsets":
            self.descriptor = _MultiHeadDeepSets(args)
            rep_dim = self.descriptor.output_dim
        elif self.arch == "slot_query":
            self.slot_critic_encoder = str(getattr(args, "mec_slot_critic_encoder", "separate"))
            if self.slot_critic_encoder == "separate":
                self.encoder = _SlotEncoder(args, UAV_STATE_DIM)
            elif self.slot_critic_encoder in {"actor_detached", "shared_grad"}:
                if actor_encoder is None:
                    raise ValueError("slot_query critic sharing requires an actor encoder")
                self.encoder = actor_encoder
            else:
                raise ValueError("mec_slot_critic_encoder must be separate|actor_detached|shared_grad")
            self.readout = _SlotReadout(args, PUBLIC_STATE_DIM)
            rep_dim = 2 * self.hidden_size
        else:
            raise ValueError("MEC supports --mec_policy_arch mean|flat|hotspot_pool|anchor_pool|grid_pool|csd_pool|slot_query|mhd_deepsets")
        input_dim = PUBLIC_STATE_DIM + rep_dim
        if self.arch == "mhd_deepsets":
            self.base = _SplitMLPBase(args, [PUBLIC_STATE_DIM, rep_dim])
        else:
            self.base = _mlp(args, rep_dim if self.arch == "slot_query" else input_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]

        def init_(module):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(PopArt(self.hidden_size, 1, device=device) if self._use_popart else nn.Linear(self.hidden_size, 1))
        self.to(device)

    def _features(self, cent_obs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        public = cent_obs[:, :PUBLIC_STATE_DIM]
        uavs = cent_obs[:, PUBLIC_STATE_DIM:].reshape(-1, self.num_uavs, UAV_STATE_DIM)
        if self.arch == "mean":
            rep = uavs.mean(dim=1)
        elif self.arch == "flat":
            rep = uavs.reshape(uavs.shape[0], -1)
        elif self.arch == "slot_query":
            if self.slot_critic_encoder == "actor_detached":
                with torch.no_grad():
                    slots = self.encoder(uavs)
            else:
                slots = self.encoder(uavs)
            return self.base(self.readout(public[:, None, :], slots).squeeze(1))
        elif self.arch == "mhd_deepsets":
            rep = self.descriptor(public, uavs)
            return self.base.forward_parts([public, rep])
        else:
            rep = _pooled_rep(public, uavs, self.arch)
        return self.base(torch.cat([public, rep], dim=-1))

    def forward(self, cent_obs, rnn_states, masks):
        rnn_states = check(rnn_states).to(**self.tpdv)
        return self.v_out(self._features(cent_obs)), rnn_states
