"""
Redistribute predicted points toward uniform surface coverage -- the
inference-time attack on CD's second term.

Why this angle: four attempts to teach the network better point
DISTRIBUTION (uniformity, coverage, tangent penalty, CD-at-the-head) all
failed to move the CD score, and --exact_score's big loss showed the
tangential scramble is unlearnable per point. But CD does not care which
point went where -- it only compares SETS. So the remaining route is to fix
the set directly: gather more surface-hugging candidates than we need, then
keep a subset chosen for even coverage.

Candidates come free: predictions of the SAME clouds at different
--sigma_scale (or different checkpoints) all lie near the surface but are
scrambled differently. Their union covers the surface more completely than
any single one. From that union this script keeps N points, preferring
points in sparse regions (weight = k-NN distance ^ power), which drains the
clumps and fills the holes -- exactly the failure the score loss cannot see.

Selection is Gumbel-top-N weighted sampling without replacement, after an
optional voxel dedup that removes near-twin candidates (different sigma
runs move the same input point almost identically when the model is
confident, and twins would otherwise soak up picks).

Usage (three sigma variants of the same predictions):
    python resample_uniform.py \
        --pred_roots results/jb3_bare_s1.0 results/jb3_bare_s1.1 results/jb3_bare_s1.2 \
        --out_root results/jb3_resampled \
        --power 2 --k 12
Then evaluate results/jb3_resampled as usual.
"""
import argparse
import os

import numpy as np
from scipy.spatial import cKDTree


def find_keys(root, name):
    keys = []
    for dirpath, _, files in os.walk(root):
        if name in files:
            keys.append(os.path.relpath(dirpath, root))
    return sorted(keys)


def voxel_dedup(pts, cell):
    """Keep one candidate per (fine) voxel. Removes near-twins without
    touching genuine structure: cell is a fraction of the point spacing."""
    ij = np.floor(pts / cell).astype(np.int64)
    # hash the 3 integer coords into one key
    key = (ij[:, 0] * 73856093) ^ (ij[:, 1] * 19349663) ^ (ij[:, 2] * 83492791)
    _, keep = np.unique(key, return_index=True)
    return pts[np.sort(keep)]


def fps_select(cands, n_out, start):
    """Farthest point sampling: greedily pick the candidate farthest from
    everything picked so far. Unlike independent weighted draws, this
    ENFORCES spacing -- no two selected points can be close, because a close
    point is by construction never the farthest one. That is the blue-noise
    property the weighted sampler lacked (it collapsed p5 spacing 4x by
    letting neighbours win together)."""
    N = cands.shape[0]
    dmin = np.full(N, np.inf)
    sel = np.empty(n_out, dtype=np.int64)
    cur = int(start)
    for i in range(n_out):
        sel[i] = cur
        diff = cands - cands[cur]
        d = np.einsum('ij,ij->i', diff, diff)
        np.minimum(dmin, d, out=dmin)
        dmin[cur] = -1.0          # never re-pick
        cur = int(np.argmax(dmin))
    return sel


def main(args):
    roots = args.pred_roots
    keysets = [set(find_keys(r, args.pred_filename)) for r in roots]
    keys = sorted(set.intersection(*keysets))
    if not keys:
        raise SystemExit('no common keys across the given --pred_roots')

    rng = np.random.default_rng(args.seed)
    for key in keys:
        sets = [np.load(os.path.join(r, key, args.pred_filename)).astype(np.float64)
                for r in roots]
        n_out = sets[0].shape[0] if args.n_points <= 0 else args.n_points
        cands = np.concatenate(sets, axis=0)

        # median NN spacing of ONE set = the natural length scale
        d1 = cKDTree(sets[0]).query(sets[0], k=2)[0][:, 1]
        spacing = float(np.median(d1))

        if args.dedup_frac > 0:
            cands = voxel_dedup(cands, cell=args.dedup_frac * spacing)
        if cands.shape[0] < n_out:
            raise SystemExit(f'{key}: dedup left {cands.shape[0]} < {n_out} '
                             f'candidates; lower --dedup_frac')

        dk = cKDTree(cands).query(cands, k=args.k + 1)[0][:, -1]

        if args.method == 'fps':
            # Outlier guard FIRST: FPS pursues the farthest point, which is
            # precisely an off-surface straggler if any survive. Drop
            # candidates whose neighbourhood is abnormally empty.
            keep = dk <= args.outlier_mult * np.median(dk)
            trimmed = cands[keep]
            if trimmed.shape[0] < n_out:
                trimmed = cands          # trim too aggressive; fall back
            # start from the DENSEST candidate: a guaranteed-inlier anchor
            start = int(np.argmin(dk[keep] if trimmed is not cands else dk))
            idx = fps_select(trimmed, n_out, start)
            out = trimmed[idx].astype(np.float32)
        else:
            # sparse regions get big weights: kNN distance ^ power. power=2
            # matches the surface's intrinsic dimension -- weight ~ the area
            # a candidate "owns", which uniform coverage equalises. Kept for
            # the record: measured CD 61.69 vs FPS's motivation -- static
            # weights cannot stop neighbours from winning together.
            w = np.maximum(dk, 1e-12) ** args.power
            g = rng.gumbel(size=cands.shape[0])
            idx = np.argpartition(-(np.log(w) + g), n_out - 1)[:n_out]
            out = cands[idx].astype(np.float32)

        out_path = os.path.join(args.out_root, key)
        os.makedirs(out_path, exist_ok=True)
        np.save(os.path.join(out_path, args.pred_filename), out)

        d1o = cKDTree(out).query(out, k=2)[0][:, 1]
        print(f'{key}: {cands.shape[0]} cands -> {n_out}   '
              f'spacing p5/p50/p95  in {np.percentile(d1, 5):.5f}/'
              f'{spacing:.5f}/{np.percentile(d1, 95):.5f}  ->  '
              f'out {np.percentile(d1o, 5):.5f}/{np.median(d1o):.5f}/'
              f'{np.percentile(d1o, 95):.5f}')

    print(f'\nwrote {len(keys)} clouds under {args.out_root}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pred_roots', nargs='+', required=True,
                   help='2+ prediction dirs of the SAME clouds (different '
                        'sigma_scale, checkpoints, or jitters)')
    p.add_argument('--out_root', required=True)
    p.add_argument('--pred_filename', default='denoised.npy')
    p.add_argument('--n_points', type=int, default=0,
                   help='output count; 0 = match the first pred set')
    p.add_argument('--k', type=int, default=12,
                   help='which NN distance defines local sparsity; must '
                        'exceed the near-twin count (~len(pred_roots))')
    p.add_argument('--method', choices=['fps', 'weighted'], default='fps',
                   help='fps = farthest point sampling (blue noise, enforced '
                        'spacing); weighted = the measured-and-rejected '
                        'independent sampler, kept for comparison')
    p.add_argument('--outlier_mult', type=float, default=3.0,
                   help='fps only: drop candidates whose kNN distance exceeds '
                        'this multiple of the median before selecting')
    p.add_argument('--power', type=float, default=2.0,
                   help='weighted only: weight = kNN_dist^power')
    p.add_argument('--dedup_frac', type=float, default=0.3,
                   help='voxel dedup cell as a fraction of the median NN '
                        'spacing; 0 disables')
    p.add_argument('--seed', type=int, default=2024)
    args = p.parse_args()
    main(args)
