"""
Pure-Jittor replacements for:
  - pointops.furthestsampling / pointops.queryandgroup / pointops.interpolation
    (originally custom CUDA extensions operating on "offset" (flattened-batch) format)
  - pytorch3d.ops.knn_points
  - the ratio-based farthest_point_sampling from models/utils.py (torch_cluster.fps)

These are written as exact, brute-force implementations (no custom CUDA kernels),
so they are slower than the originals but numerically equivalent in behavior.

Offset format recap (as used throughout blocks.py):
  p, x are (N_total, 3) / (N_total, C) tensors where several point clouds of
  possibly different sizes are concatenated along dim 0.
  o is a 1D tensor of CUMULATIVE counts, e.g. o = [1000, 2000, 3000] means
  batch 0 = p[0:1000], batch 1 = p[1000:2000], batch 2 = p[2000:3000].
"""

import jittor as jt
import math


def _offsets_to_bounds(o):
    """Convert a cumulative-offset tensor/list into a list of (start, end) pairs."""
    if isinstance(o, jt.Var):
        o = o.numpy().tolist()
    else:
        o = list(o)
    bounds = []
    prev = 0
    for end in o:
        end = int(end)
        bounds.append((prev, end))
        prev = end
    return bounds


def _pairwise_sqdist(a, b):
    """a: (Na,3), b: (Nb,3) -> (Na,Nb) squared euclidean distances."""
    a2 = (a * a).sum(dim=-1, keepdims=True)          # (Na,1)
    b2 = (b * b).sum(dim=-1, keepdims=True).transpose(1, 0)  # (1,Nb)
    ab = jt.matmul(a, b.transpose(1, 0))              # (Na,Nb)
    return a2 + b2 - 2 * ab


def furthestsampling(p, o, n_o):
    """
    Exact iterative farthest-point sampling, per-batch, in offset format.

    Args:
        p:   (N_total, 3) jt.Var
        o:   cumulative offsets of the input, length B
        n_o: cumulative offsets of the desired output, length B
    Returns:
        idx: (M_total,) LongVar of indices into p (flat, i.e. already including
             the per-batch start offset), where M_total = n_o[-1].
    """
    in_bounds = _offsets_to_bounds(o)
    out_bounds = _offsets_to_bounds(n_o)

    all_idx = []
    for (start, end), (out_start, out_end) in zip(in_bounds, out_bounds):
        n_pts = end - start
        n_sample = out_end - out_start
        pts = p[start:end]  # (n_pts, 3)

        if n_sample <= 0:
            continue

        selected = jt.zeros((n_sample,), dtype='int32')
        dist = jt.full((n_pts,), 1e10)
        farthest = 0  # deterministic start point, matches random_start=False behavior

        for i in range(n_sample):
            selected[i] = farthest
            centroid = pts[farthest:farthest + 1, :]  # (1,3)
            d = ((pts - centroid) ** 2).sum(dim=-1)    # (n_pts,)
            dist = jt.minimum(dist, d)
            farthest = int(jt.argmax(dist, dim=0)[0].item())

        all_idx.append(selected + start)

    return jt.concat(all_idx, dim=0) if all_idx else jt.zeros((0,), dtype='int32')


def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset,
                   use_xyz=True, return_index=False):
    """
    For each query point in new_xyz, gather its `nsample` nearest neighbors
    (restricted to the same batch element via offset/new_offset) from xyz,
    and return [relative_xyz, features] concatenated on the last dim.

    Shapes:
        xyz:     (N_total, 3)
        new_xyz: (M_total, 3)
        feat:    (N_total, C)
    Returns:
        grouped: (M_total, nsample, 3 + C) if use_xyz else (M_total, nsample, C)
        idx_out: (M_total, nsample) LongVar of neighbor indices into xyz (flat)
    """
    xyz_bounds = _offsets_to_bounds(offset)
    query_bounds = _offsets_to_bounds(new_offset)

    grouped_list = []
    idx_list = []

    for (xs, xe), (qs, qe) in zip(xyz_bounds, query_bounds):
        xyz_b = xyz[xs:xe]        # (n_b, 3)
        query_b = new_xyz[qs:qe]  # (m_b, 3)
        feat_b = feat[xs:xe]      # (n_b, C)

        sqd = _pairwise_sqdist(query_b, xyz_b)  # (m_b, n_b)
        # nearest `nsample` neighbors (ascending distance)
        nn_idx = jt.argsort(sqd, dim=-1)[0][:, :nsample]  # (m_b, nsample) local idx

        grouped_xyz = xyz_b[nn_idx]                   # (m_b, nsample, 3)
        grouped_xyz = grouped_xyz - query_b.unsqueeze(1)  # relative coords
        grouped_feat = feat_b[nn_idx]                  # (m_b, nsample, C)

        if use_xyz:
            grouped = jt.concat([grouped_xyz, grouped_feat], dim=-1)
        else:
            grouped = grouped_feat

        grouped_list.append(grouped)
        idx_list.append(nn_idx + xs)

    grouped_out = jt.concat(grouped_list, dim=0)
    idx_out = jt.concat(idx_list, dim=0)

    if return_index:
        return grouped_out, idx_out
    return grouped_out, None


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """
    Three(or k)-nearest-neighbor inverse-distance-weighted feature interpolation
    from a sparse point set (xyz, feat) onto a dense point set (new_xyz).

    Args:
        xyz:     (N_total, 3) sparse/source points
        new_xyz: (M_total, 3) dense/target points
        feat:    (N_total, C) sparse/source features
        offset:     cumulative offsets for xyz  (source)
        new_offset: cumulative offsets for new_xyz (target)
        k: number of neighbors to interpolate from
    Returns:
        (M_total, C) interpolated features at new_xyz locations
    """
    src_bounds = _offsets_to_bounds(offset)
    dst_bounds = _offsets_to_bounds(new_offset)

    out_list = []
    eps = 1e-8

    for (ss, se), (ds, de) in zip(src_bounds, dst_bounds):
        src_xyz = xyz[ss:se]        # (n_b,3)
        dst_xyz = new_xyz[ds:de]    # (m_b,3)
        src_feat = feat[ss:se]      # (n_b,C)

        sqd = _pairwise_sqdist(dst_xyz, src_xyz)     # (m_b, n_b)
        knn_dist_sorted_idx = jt.argsort(sqd, dim=-1)[0][:, :k]     # (m_b,k)
        knn_dist = jt.argsort(sqd, dim=-1)[1][:, :k]                # sorted sq-dists (m_b,k)

        weight = 1.0 / (knn_dist + eps)
        weight = weight / weight.sum(dim=-1, keepdims=True)  # (m_b,k)

        gathered = src_feat[knn_dist_sorted_idx]              # (m_b,k,C)
        interpolated = (gathered * weight.unsqueeze(-1)).sum(dim=1)  # (m_b,C)
        out_list.append(interpolated)

    return jt.concat(out_list, dim=0)


def knn_points(query, ref, K=1, return_nn=False):
    """
    Drop-in equivalent of pytorch3d.ops.knn_points for batched tensors.

    Args:
        query: (B, Nq, 3)
        ref:   (B, Nr, 3)
        K: number of neighbors
        return_nn: also return gathered neighbor coordinates
    Returns:
        dists: (B, Nq, K) squared distances, ascending
        idx:   (B, Nq, K) indices into ref
        nn:    (B, Nq, K, 3) neighbor coordinates, or None if return_nn=False
    """
    B = query.shape[0]
    dists_list, idx_list, nn_list = [], [], []

    for b in range(B):
        sqd = _pairwise_sqdist(query[b], ref[b])  # (Nq, Nr)
        order = jt.argsort(sqd, dim=-1)
        idx_b = order[0][:, :K]     # (Nq,K)
        dist_b = order[1][:, :K]    # (Nq,K)
        dists_list.append(dist_b.unsqueeze(0))
        idx_list.append(idx_b.unsqueeze(0))
        if return_nn:
            nn_list.append(ref[b][idx_b].unsqueeze(0))

    dists = jt.concat(dists_list, dim=0)
    idx = jt.concat(idx_list, dim=0)
    nn = jt.concat(nn_list, dim=0) if return_nn else None
    return dists, idx, nn


def farthest_point_sampling(pcls, num_pnts):
    """
    Exact per-batch FPS, replacing the original ratio-based torch_cluster.fps call
    (which only approximately returns num_pnts points). This version returns
    exactly num_pnts points every time.

    Args:
        pcls: (B, N, 3)
        num_pnts: target number of points
    Returns:
        sampled: (B, num_pnts, 3)
        indices: list of length B, each a (num_pnts,) LongVar of local indices
    """
    B, N, _ = pcls.shape
    sampled = []
    indices = []

    for b in range(B):
        pts = pcls[b]  # (N,3)
        selected = jt.zeros((num_pnts,), dtype='int32')
        dist = jt.full((N,), 1e10)
        farthest = 0

        for i in range(num_pnts):
            selected[i] = farthest
            centroid = pts[farthest:farthest + 1, :]
            d = ((pts - centroid) ** 2).sum(dim=-1)
            dist = jt.minimum(dist, d)
            farthest = int(jt.argmax(dist, dim=0)[0].item())

        indices.append(selected)
        sampled.append(pts[selected].unsqueeze(0))

    sampled = jt.concat(sampled, dim=0)
    return sampled, indices
