'''
==============================================================

    0-------------------------------0
    |       Loss Functions          |
    0-------------------------------0

==============================================================

    Compute chamfer-distance-based InfoCD loss (Jittor version)

    The original imported chamfer_3DDist from the Chamfer3D CUDA
    extension. Here it is replaced with a pure-Jittor implementation
    based on brute-force KNN (K=1 in both directions), returning the
    same (dist1, dist2, idx1, idx2) quadruple: squared distances and
    nearest-neighbor indices in each direction.

==============================================================
'''

import jittor as jt

from .pointops_jt import knn_points

MIN_NORM = 1e-15


def chamfer_dist(p1, p2):
    """
    p1: (B, N, 3), p2: (B, M, 3)
    Returns:
        dist1: (B, N) squared distance from each p1 point to nearest p2 point
        dist2: (B, M) squared distance from each p2 point to nearest p1 point
        idx1:  (B, N) index of that nearest p2 point
        idx2:  (B, M) index of that nearest p1 point
    """
    d1, i1, _ = knn_points(p1, p2, K=1)
    d2, i2, _ = knn_points(p2, p1, K=1)
    return d1[:, :, 0], d2[:, :, 0], i1[:, :, 0], i2[:, :, 0]


def calc_cd_like_InfoV2(p1, p2):
    dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
    dist1 = jt.clamp(dist1, min_v=1e-9)
    dist2 = jt.clamp(dist2, min_v=1e-9)
    d1 = jt.sqrt(dist1)
    d2 = jt.sqrt(dist2)

    distances1 = -jt.log(jt.exp(-0.5 * d1) /
                          (jt.sum(jt.exp(-0.5 * d1) + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7)
    distances2 = -jt.log(jt.exp(-0.5 * d2) /
                          (jt.sum(jt.exp(-0.5 * d2) + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7)

    return (jt.sum(distances1) + jt.sum(distances2)) / (2 * p1.shape[0])
