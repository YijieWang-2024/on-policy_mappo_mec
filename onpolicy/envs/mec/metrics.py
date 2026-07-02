"""Demand-matching metrics for the MEC env (spec section 6: W1(mu_UAV, mu_demand)).

True W1 between a K-atom UAV measure and a gridded demand measure is a transport
LP. We report the *semi-discrete 1-NN transport cost* (demand mass -> nearest
UAV), which is the natural, hyperparameter-free "are the UAVs sitting where the
demand is" distance and an upper bound on balanced W1. Lower = the swarm spatial
distribution matches the demand density. Reported in metres.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAVE_KDTREE = True
except Exception:  # pragma: no cover
    _HAVE_KDTREE = False


def demand_matching_w1(uav_xy: np.ndarray, grid_xy: np.ndarray,
                       demand_mass: np.ndarray) -> float:
    """Mass-weighted mean distance from demand to the nearest UAV (semi-discrete W1).

    uav_xy:      (K, 2) UAV positions [m].
    grid_xy:     (G, 2) quadrature cell centers [m] (env.grid_xy).
    demand_mass: (G,)   nonneg demand weight per cell (e.g. active density * area).
    """
    uav_xy = np.asarray(uav_xy, float)
    grid_xy = np.asarray(grid_xy, float)
    w = np.asarray(demand_mass, float)
    total = float(w.sum())
    if total <= 0.0 or uav_xy.shape[0] == 0:
        return 0.0
    if _HAVE_KDTREE:
        nn_dist, _ = cKDTree(uav_xy).query(grid_xy, k=1)
    else:  # pragma: no cover
        d = np.linalg.norm(grid_xy[:, None, :] - uav_xy[None, :, :], axis=2)
        nn_dist = d.min(axis=1)
    return float(np.sum(w * nn_dist) / total)
