"""Shared-policy runner for the MEC environment.

Reuses the MPE shared runner wholesale (Gymnasium termination semantics,
truncation bootstrap, GAE, logging) and only adds D4: a Box action passthrough.
MEC actions are continuous (`[-1,1]^3` per agent); the env does the projection,
so the runner forwards raw actions unchanged instead of one-hot expanding.
"""

from __future__ import annotations

import numpy as np

from onpolicy.envs.mec.observation import repeat_team_state
from onpolicy.runner.shared.mpe_runner import MPERunner


class MECRunner(MPERunner):
    def _share_obs(self, obs):
        if not self.use_centralized_V:
            return obs
        return repeat_team_state(obs)

    def _actions_to_env(self, actions, envs):
        action_space = envs.action_space[0]
        if action_space.__class__.__name__ == "Box":
            return actions  # [threads, agents, act_dim] -> env projects internally
        return super()._actions_to_env(actions, envs)

    def log_env(self, env_infos, total_num_steps):
        """Also surface team MEC metrics (carried on the major agent's info slot)."""
        is_training_snapshot = any(
            key.startswith("agent") for key in env_infos
        )
        if not is_training_snapshot:
            # eval() currently supplies only its aggregate reward. Reusing
            # _last_infos here would silently print the most recent training
            # rollout as though it came from validation.
            return super().log_env(env_infos, total_num_steps)

        keys = (
            "training_cost", "src_cost", "ovf_cost", "queue_cost",
            "energy_cost", "accepted", "offloaded", "overflow", "U_src",
            "access_utilization", "backhaul_utilization",
            "uav_compute_utilization", "hub_compute_utilization",
            "source_outside", "source_capacity",
            "hotspot_offered", "hotspot_accepted", "hotspot_source",
            "background_offered", "background_accepted", "background_source",
            "eta_p05", "eta_p50", "eta_p95", "eta_served", "eta_all",
            "n_hotspot_uav", "n_background_uav", "hub_to_hotspot", "w1",
        )
        for key in keys:
            vals = []
            for info in getattr(self, "_last_infos", []):
                major = info.get("agent_infos", [{}])[0]
                if key in major and major[key] is not None:
                    vals.append(major[key])
            if vals:
                env_infos[f"mec/{key}"] = vals
        self._print_mec_summary(env_infos)
        super().log_env(env_infos, total_num_steps)

    @staticmethod
    def _print_mec_summary(env_infos):
        """One-line console trace of the validation metrics (last-step, thread-mean).

        Lets a plain log tail track whether `accepted` collapses and whether `w1`
        drops over training without opening TensorBoard. These are last-step
        snapshots; the deterministic episode-mean truth comes from eval_mec.py.
        """
        def m(key):
            vals = env_infos.get(f"mec/{key}")
            return float(np.mean(vals)) if vals else float("nan")

        acc, usrc, total = m("accepted"), m("U_src"), m("training_cost")
        denom = acc + usrc
        accept_rate = 100.0 * acc / denom if denom > 0 else float("nan")
        share = lambda k: 100.0 * m(k) / total if total else float("nan")
        print(
            f"  [mec] accept={accept_rate:4.1f}%  accepted={acc/1e6:5.1f}  "
            f"ovf={m('overflow')/1e6:4.1f}  U_src={usrc/1e6:5.1f} "
            f"(out={m('source_outside')/1e6:4.1f}, cap={m('source_capacity')/1e6:4.1f})  "
            f"n_hot={m('n_hotspot_uav'):4.1f}  hub={m('hub_to_hotspot'):5.0f}m  "
            f"W1={m('w1'):6.0f}m  | shares src={share('src_cost'):3.0f}% "
            f"ovf={share('ovf_cost'):3.0f}% q={share('queue_cost'):3.0f}% "
            f"e={share('energy_cost'):3.0f}%"
        )

    def insert(self, data):
        # stash the latest infos so log_env can read team metrics
        self._last_infos = data[4]
        super().insert(data)
