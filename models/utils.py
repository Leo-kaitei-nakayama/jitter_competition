import math
import numpy as np
import jittor as jt

from .pointops_jt import knn_points, farthest_point_sampling  # re-export FPS


def entropy_from_histogram(hist):
    """hist: 1D jt.Var. Entropy of a (already >0-filtered) histogram."""
    hist = hist[hist > 0]
    p = hist / jt.sum(hist)
    return -jt.sum(p * jt.log(p))


def get_entropy_B(point_clouds, voxel_size=0.01):
    """
    Voxel-density entropy per point cloud, mirroring the original torch version
    (dynamic voxel size, unique-voxel counting, 10-bin histogram).

    Computed via numpy internally: no gradients ever flow through this in the
    original either (it is only used to build supervision targets).

    Args:
        point_clouds: (B, N, 3) jt.Var
    Returns:
        (B,) jt.Var of entropies
    """
    entropies = []
    pcs = point_clouds.numpy()

    for pc in pcs:
        vsize = 0.001 * 100000 / pc.shape[0]  # dynamic voxel size (as original)
        min_bound = pc.min(axis=0)
        indices = np.floor((pc - min_bound) / vsize).astype(np.int64)
        _, inverse = np.unique(indices, axis=0, return_inverse=True)
        voxel_densities = np.bincount(inverse).astype(np.float32)

        vmax = voxel_densities.max()
        hist, _ = np.histogram(voxel_densities, bins=10, range=(0.0, float(vmax)))
        hist = hist.astype(np.float32)
        hist[hist == 0] = 1.0
        hist = hist / hist.sum()

        h = hist[hist > 0]
        p = h / h.sum()
        ent = float(-(p * np.log(p)).sum())
        entropies.append(ent)

    return jt.array(np.array(entropies, dtype=np.float32))


def normalize_sphere(pc, radius=1.0):
    """pc: (B, N, 3)"""
    p_max = pc.max(dim=-2, keepdims=True)
    p_min = pc.min(dim=-2, keepdims=True)
    center = (p_max + p_min) / 2  # (B, 1, 3)
    pc = pc - center
    scale = jt.sqrt((pc ** 2).sum(dim=-1, keepdims=True)).max(dim=-2, keepdims=True) / radius
    pc = pc / scale
    return pc, center, scale


def normalize_std(pc, std=1.0):
    """pc: (B, N, 3)"""
    center = pc.mean(dim=-2, keepdims=True)
    pc = pc - center
    scale = pc.reshape(pc.shape[0], -1).std(dim=-1).reshape(pc.shape[0], 1, 1) / std
    pc = pc / scale
    return pc, center, scale


def normalize_pcl(pc, center, scale):
    return (pc - center) / scale


def denormalize_pcl(pc, center, scale):
    return pc * scale + center


def chamfer_distance(gen, ref, batch_reduction='mean', point_reduction='mean'):
    """
    Pure-Jittor bidirectional Chamfer distance (squared), matching
    pytorch3d.loss.chamfer_distance's return convention: (loss, None).

    gen, ref: (B, N, 3) / (B, M, 3)
    """
    d_ab, _, _ = knn_points(gen, ref, K=1)   # (B, N, 1) sq dists
    d_ba, _, _ = knn_points(ref, gen, K=1)   # (B, M, 1)

    d_ab = d_ab[:, :, 0]
    d_ba = d_ba[:, :, 0]

    if point_reduction == 'mean':
        cham = d_ab.mean(dim=1) + d_ba.mean(dim=1)  # (B,)
    else:  # 'sum'
        cham = d_ab.sum(dim=1) + d_ba.sum(dim=1)

    if batch_reduction == 'mean':
        return cham.mean(), None
    elif batch_reduction == 'sum':
        return cham.sum(), None
    return cham, None


def chamfer_distance_unit_sphere(gen, ref, batch_reduction='mean', point_reduction='mean'):
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)
    return chamfer_distance(gen, ref, batch_reduction=batch_reduction, point_reduction=point_reduction)


def hausdorff_distance_unit_sphere(gen, ref):
    """
    gen, ref: (B, N, 3) -> (B,)
    """
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)

    dists_ab, _, _ = knn_points(ref, gen, K=1)
    dists_ab = dists_ab[:, :, 0].max(dim=1, keepdims=True)  # (B, 1)

    dists_ba, _, _ = knn_points(gen, ref, K=1)
    dists_ba = dists_ba[:, :, 0].max(dim=1, keepdims=True)  # (B, 1)

    dists_hausdorff = jt.concat([dists_ab, dists_ba], dim=1).max(dim=1)
    return dists_hausdorff


# ---------------------------------------------------------------------------
# Point-to-mesh distance (used by Evaluate.py). numpy implementation of exact
# point-triangle squared distance; no gradients required for evaluation.
# ---------------------------------------------------------------------------

def _point_triangle_sqdist_np(points, tri_a, tri_b, tri_c):
    """
    points: (P, 3); tri_a/b/c: (T, 3) triangle vertices.
    Returns (P,) min squared distance from each point to the closest triangle.
    Vectorized over triangles per point-chunk to bound memory.
    """
    P = points.shape[0]
    out = np.full((P,), np.inf, dtype=np.float64)

    ab = tri_b - tri_a  # (T,3)
    ac = tri_c - tri_a

    chunk = max(1, int(2e7 // max(1, tri_a.shape[0])))
    for s in range(0, P, chunk):
        p = points[s:s + chunk]  # (p,3)
        ap = p[:, None, :] - tri_a[None, :, :]     # (p,T,3)

        d1 = (ab[None] * ap).sum(-1)               # (p,T)
        d2 = (ac[None] * ap).sum(-1)

        bp = p[:, None, :] - tri_b[None, :, :]
        d3 = (ab[None] * bp).sum(-1)
        d4 = (ac[None] * bp).sum(-1)

        cp = p[:, None, :] - tri_c[None, :, :]
        d5 = (ab[None] * cp).sum(-1)
        d6 = (ac[None] * cp).sum(-1)

        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2

        denom_uv = (vc + vb + va)
        denom_uv = np.where(np.abs(denom_uv) < 1e-30, 1e-30, denom_uv)
        v = vb / denom_uv
        w = vc / denom_uv
        # closest point on the triangle's plane, clamped by region tests below
        closest = tri_a[None] + v[..., None] * ab[None] + w[..., None] * ac[None]

        # Region clamping (standard Ericson point-triangle algorithm)
        # vertex A region
        mask = (d1 <= 0) & (d2 <= 0)
        closest = np.where(mask[..., None], tri_a[None].repeat(p.shape[0], axis=0), closest)
        # vertex B region
        mask = (d3 >= 0) & (d4 <= d3)
        closest = np.where(mask[..., None], tri_b[None].repeat(p.shape[0], axis=0), closest)
        # vertex C region
        mask = (d6 >= 0) & (d5 <= d6)
        closest = np.where(mask[..., None], tri_c[None].repeat(p.shape[0], axis=0), closest)
        # edge AB region
        mask = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = np.clip(d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3), 0, 1)
        edge_pt = tri_a[None] + t[..., None] * ab[None]
        closest = np.where(mask[..., None], edge_pt, closest)
        # edge AC region
        mask = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = np.clip(d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6), 0, 1)
        edge_pt = tri_a[None] + t[..., None] * ac[None]
        closest = np.where(mask[..., None], edge_pt, closest)
        # edge BC region
        mask = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t = np.clip((d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-30, 1e-30,
                                          (d4 - d3) + (d5 - d6)), 0, 1)
        edge_pt = tri_b[None] + t[..., None] * (tri_c - tri_b)[None]
        closest = np.where(mask[..., None], edge_pt, closest)

        sqd = ((p[:, None, :] - closest) ** 2).sum(-1)  # (p,T)
        out[s:s + chunk] = sqd.min(axis=1)

    return out


def point_mesh_bidir_distance_single_unit_sphere(pcl, verts, faces):
    """
    Bidirectional point<->mesh distance after unit-sphere normalization,
    approximating pytorch3d's point_mesh_face_distance (point->face mean +
    face-sample->point mean).

    pcl: (N,3) jt.Var; verts: (M,3) jt.Var; faces: (T,3) int
    Returns scalar (python float wrapped in jt.Var-compatible use).
    """
    assert pcl.ndim == 2 and verts.ndim == 2 and faces.ndim == 2

    verts_n, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts_n = verts_n[0]
    pcl_n = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)[0]

    v = verts_n.numpy().astype(np.float64)
    f = faces.numpy().astype(np.int64)
    p = pcl_n.numpy().astype(np.float64)

    tri_a, tri_b, tri_c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]

    # point -> face
    p2f = _point_triangle_sqdist_np(p, tri_a, tri_b, tri_c).mean()

    # face -> point (sample the face centroids against the point cloud,
    # a standard approximation of the reverse term)
    centroids = (tri_a + tri_b + tri_c) / 3.0
    d = ((centroids[:, None, :] - p[None, :, :]) ** 2).sum(-1).min(axis=1)
    f2p = d.mean()

    return jt.array(np.float32(p2f + f2p))


def pointwise_p2m_distance_normalized(pcl, verts, faces):
    """Per-point squared point->mesh distance after unit-sphere normalization."""
    assert pcl.ndim == 2 and verts.ndim == 2 and faces.ndim == 2

    verts_n, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts_n = verts_n[0]
    pcl_n = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)[0]

    v = verts_n.numpy().astype(np.float64)
    f = faces.numpy().astype(np.int64)
    p = pcl_n.numpy().astype(np.float64)

    tri_a, tri_b, tri_c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    sqd = _point_triangle_sqdist_np(p, tri_a, tri_b, tri_c)
    return jt.array(sqd.astype(np.float32))
