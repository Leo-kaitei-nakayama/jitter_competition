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
import numpy as np
import math


def _fps_batched(pts, n_sample):
    """
    Exact farthest-point sampling, vectorized over the batch.

    Produces byte-identical output to the per-segment Python loop it replaces
    (deterministic start at index 0, ties broken by argmax), with one change:
    the running `farthest` index stays on the device as a Var instead of being
    pulled to the host with .item() on every step.

    That single change is what matters for speed. The old loop issued one
    blocking GPU sync per sampled point, and Downsampling calls FPS once per
    encoder block per batch element -- roughly 16k syncs per forward pass at the
    default 1000-point patches and stride_list=[4,3,2,1] with batch 8. Sync
    latency, not arithmetic, dominated the step time, which is also why adding
    GPUs did not speed training up.

    Args:
        pts: (B, N, 3)
        n_sample: number of points to select per batch element
    Returns:
        (B, n_sample) int32 Var of indices into the N axis
    """
    B, N, _ = pts.shape
    if n_sample <= 0:
        return jt.zeros((B, 0), dtype='int32')

    batch_ar = jt.arange(B)                       # (B,)
    dist = jt.full((B, N), 1e10)
    farthest = jt.zeros((B,), dtype='int32')      # deterministic start point
    picked = []

    for _ in range(n_sample):
        picked.append(farthest)
        centroid = pts[batch_ar, farthest].unsqueeze(1)   # (B, 1, 3), device-side gather
        d = ((pts - centroid) ** 2).sum(dim=-1)           # (B, N)
        dist = jt.minimum(dist, d)
        # cast keeps every entry of `picked` the same dtype for the stack below,
        # regardless of what width argmax returns
        farthest = jt.argmax(dist, dim=1)[0].int32()      # (B,), never leaves the device

    return jt.stack(picked, dim=1)


def _fps_reference(pts, n_sample):
    """
    The original per-point loop, kept only so bench_fps.py can prove that
    _fps_batched returns the same indices. Not used in training or inference.

    pts: (N, 3) -> (n_sample,) int32 Var of local indices.
    """
    n_pts = pts.shape[0]
    selected = jt.zeros((n_sample,), dtype='int32')
    dist = jt.full((n_pts,), 1e10)
    farthest = 0

    for i in range(n_sample):
        selected[i] = farthest
        centroid = pts[farthest:farthest + 1, :]
        d = ((pts - centroid) ** 2).sum(dim=-1)
        dist = jt.minimum(dist, d)
        farthest = int(jt.argmax(dist, dim=0)[0].item())

    return selected


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

    sizes_in = [e - s for s, e in in_bounds]
    sizes_out = [e - s for s, e in out_bounds]

    # Fast path: all segments the same size, which is always the case for the
    # offsets this codebase builds (denoiseCD makes o uniform, and Downsampling
    # derives n_o from a single `count`). Run the whole batch as one (B, N, 3).
    if (sizes_in and sizes_out[0] > 0
            and len(set(sizes_in)) == 1 and len(set(sizes_out)) == 1):
        B, N = len(sizes_in), sizes_in[0]
        pts = p[in_bounds[0][0]:in_bounds[-1][1]].reshape(B, N, 3)
        idx = _fps_batched(pts, sizes_out[0])                 # (B, M) segment-local
        starts = jt.array(
            np.array([s for s, _ in in_bounds], dtype=np.int32)).reshape(B, 1)
        return (idx + starts).reshape(-1)

    # General path: segments of differing length, one batched call each.
    all_idx = []
    for (start, end), (out_start, out_end) in zip(in_bounds, out_bounds):
        n_sample = out_end - out_start
        if n_sample <= 0:
            continue
        idx = _fps_batched(p[start:end].unsqueeze(0), n_sample)[0]
        all_idx.append(idx + start)

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

    idx = _fps_batched(pcls, num_pnts)                          # (B, num_pnts)
    batch_ar = jt.arange(B).reshape(B, 1).broadcast((B, num_pnts))
    sampled = pcls[batch_ar, idx]                               # (B, num_pnts, 3)

    # keep the list-of-Vars contract: classify.patch_based_shang does indices[0]
    indices = [idx[b] for b in range(B)]
    return sampled, indices
