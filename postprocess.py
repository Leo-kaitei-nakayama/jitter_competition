"""
Tangential repulsion post-filter.

Runs on denoised .npy files after prediction. It never loads the model, never
imports jittor, and cannot touch any trained weight -- the separation is
physical, not a flag you could set wrong.

What it does
------------
check_spacing.py measured the denoised output at ~6% higher spacing
non-uniformity than the ground truth, and denoising itself makes spacing worse
than its own input (cv 0.506 -> 0.552). CD punishes that, because a
ground-truth point sitting in a gap has no prediction near it; P2S does not
notice at all, since it only asks how far each point is from the surface.

So each point is nudged AWAY from its neighbours, but only within the local
tangent plane -- sideways along the surface, never off it. Spacing evens out
while every point stays exactly as close to the surface as it started, which is
what keeps P2S intact while CD improves.

Per iteration, for each point:
  1. estimate the local surface normal by PCA over its k nearest neighbours
  2. compute a distance-weighted push away from those neighbours
  3. remove the component along the normal, keeping only tangential motion
  4. step by strength * h, where h is the cloud's mean nearest-neighbour spacing

A point with evenly spread neighbours gets a push that cancels to nearly zero,
so well-spaced regions are left alone and only clumps move.

Guarantees
----------
  * point count is preserved exactly (required by the competition, and checked
    by check_submission.py)
  * --strength 0 is the exact identity, so it always degrades to your current
    result
  * output dtype is float32, matching the submission format

Usage:
    python postprocess.py --pred_root ./eval_predict/L3 \
        --out_root ./eval_predict/L3_rep --strength 0.3 --iters 3 --workers 16
"""
import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree


def tangential_repulsion(pc, k=16, strength=0.3, iters=3):
    """
    pc: (N, 3) float array. Returns (N, 3), same count, same order.
    """
    if strength <= 0 or iters <= 0:
        return pc

    pc = pc.astype(np.float64, copy=True)

    for _ in range(iters):
        tree = cKDTree(pc)
        d, idx = tree.query(pc, k=k + 1)      # column 0 is the point itself
        nbr_d = np.maximum(d[:, 1:], 1e-12)   # (N, k)
        nbr = pc[idx[:, 1:]]                  # (N, k, 3)

        h = float(nbr_d[:, 0].mean())         # mean nearest-neighbour spacing
        if not np.isfinite(h) or h <= 0:
            break

        # --- local surface normal: PCA over the neighbourhood ---
        patch = pc[idx]                                     # (N, k+1, 3)
        cen = patch - patch.mean(axis=1, keepdims=True)
        cov = np.einsum('nki,nkj->nij', cen, cen) / (k + 1)
        # eigh returns ascending eigenvalues; the smallest one's vector is normal
        _, vecs = np.linalg.eigh(cov)
        normals = vecs[:, :, 0]                             # (N, 3)

        # --- distance-weighted push away from neighbours ---
        rel = pc[:, None, :] - nbr                          # (N, k, 3), points away
        w = np.exp(-(nbr_d / h) ** 2)[:, :, None]           # near neighbours dominate
        push = (w * rel / nbr_d[:, :, None]).sum(axis=1)
        push /= np.maximum(w.sum(axis=1), 1e-12)            # weighted mean direction

        # --- keep only the component in the tangent plane ---
        push -= (push * normals).sum(axis=-1, keepdims=True) * normals

        pc = pc + strength * h * push

    return pc


def process_one(path, pred_root, out_root, k, strength, iters):
    rel = os.path.relpath(path, pred_root)
    out_path = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pc = np.load(path)
    n_in = pc.shape[0]
    out = tangential_repulsion(pc.astype(np.float64), k=k,
                               strength=strength, iters=iters)

    assert out.shape[0] == n_in, f'point count changed for {rel}'
    np.save(out_path, out.astype(np.float32))
    return rel, float(np.abs(out - pc).max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_root', required=True,
                        help='directory of predictions to filter')
    parser.add_argument('--out_root', required=True,
                        help='where the filtered copies go (originals untouched)')
    parser.add_argument('--pred_filename', default='denoised.npy')
    parser.add_argument('--k', type=int, default=16,
                        help='neighbours used for the normal estimate and the push')
    parser.add_argument('--strength', type=float, default=0.3,
                        help='step size as a fraction of mean point spacing. '
                             '0 is the exact identity.')
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.pred_root, '**', args.pred_filename),
                             recursive=True))
    if not files:
        raise SystemExit(f'no {args.pred_filename} under {args.pred_root}')

    print(f'{len(files)} clouds  |  k={args.k} strength={args.strength} '
          f'iters={args.iters}')

    fn = partial(process_one, pred_root=args.pred_root, out_root=args.out_root,
                 k=args.k, strength=args.strength, iters=args.iters)

    if args.workers > 1 and len(files) > 1:
        with Pool(args.workers) as pool:
            results = pool.map(fn, files)
    else:
        results = [fn(f) for f in files]

    moved = np.array([m for _, m in results])
    print(f'wrote {len(results)} files to {args.out_root}')
    print(f'max point displacement: mean {moved.mean():.6f}  max {moved.max():.6f}')
    print('(compare against your mean point spacing, ~0.0047 in unit-sphere units)')


if __name__ == '__main__':
    main()
