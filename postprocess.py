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


def jet_project(pc, k=16, strength=1.0, degree=2, ridge=1e-8):
    """
    Move each point onto a locally fitted surface patch (MLS / jet fitting).

    Denoised points scatter randomly around the true surface. Fitting a small
    polynomial patch through a point's neighbourhood averages that scatter out,
    and snapping the point onto the patch removes most of it -- which lowers
    P2S directly, and lowers CD too, since a point sitting d away from the
    surface is at least d from every ground-truth point.

    Motion is along the local normal only: the point keeps its tangential
    position, so the spacing that tangential_repulsion fixes is left intact.
    The two filters compose rather than fight.

    degree=2 fits a quadratic, which follows curvature; degree=1 fits a plane,
    which is more aggressive and flattens more. strength blends between no
    movement (0) and full projection (1).

    pc: (N, 3). Returns (N, 3), same count, same order.
    """
    if strength <= 0:
        return pc

    pc = pc.astype(np.float64, copy=False)
    _, idx = cKDTree(pc).query(pc, k=k + 1)
    patch = pc[idx]                                     # (N, k+1, 3)
    centroid = patch.mean(axis=1, keepdims=True)
    cen = patch - centroid

    # local frame: eigh gives ascending eigenvalues, so column 0 is the normal
    cov = np.einsum('nki,nkj->nij', cen, cen) / (k + 1)
    _, vecs = np.linalg.eigh(cov)
    n_ax, t1, t2 = vecs[:, :, 0], vecs[:, :, 1], vecs[:, :, 2]

    a = np.einsum('nki,ni->nk', cen, t1)                # (N, k+1)
    b = np.einsum('nki,ni->nk', cen, t2)
    c = np.einsum('nki,ni->nk', cen, n_ax)

    if degree >= 2:
        A = np.stack([np.ones_like(a), a, b, a * a, a * b, b * b], axis=-1)
    else:
        A = np.stack([np.ones_like(a), a, b], axis=-1)   # (N, k+1, m)

    # ridge-regularised normal equations, batched over points
    m = A.shape[-1]
    AtA = np.einsum('nkm,nkl->nml', A, A) + ridge * np.eye(m)
    Atc = np.einsum('nkm,nk->nm', A, c)
    # trailing axis kept explicit: numpy 2.x reads a bare (N, m) rhs as a matrix
    # rather than a stack of vectors, so this form is needed for both 1.x and 2.x
    coef = np.linalg.solve(AtA, Atc[..., None])[..., 0]   # (N, m)

    # evaluate the patch at the point's own tangential coordinates
    self_rel = pc - centroid[:, 0, :]
    sa = np.einsum('ni,ni->n', self_rel, t1)
    sb = np.einsum('ni,ni->n', self_rel, t2)
    sc = np.einsum('ni,ni->n', self_rel, n_ax)

    if degree >= 2:
        basis = np.stack([np.ones_like(sa), sa, sb, sa * sa, sa * sb, sb * sb], axis=-1)
    else:
        basis = np.stack([np.ones_like(sa), sa, sb], axis=-1)
    fitted = np.einsum('nm,nm->n', coef, basis)

    # move along the normal only, by the blended amount
    return pc + (strength * (fitted - sc))[:, None] * n_ax


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


def parse_sched(spec):
    """'0.010:0.1:1,0.015:0.5:4,999:0.6:5' -> [(limit, strength, iters), ...]"""
    bands = []
    for part in spec.split(','):
        lim, s, it = part.split(':')
        bands.append((float(lim), float(s), int(it)))
    return sorted(bands)


def process_one(path, pred_root, out_root, k, strength, iters,
                project_strength, project_degree, project_k,
                sigma_by_rel=None, sched=None):
    rel = os.path.relpath(path, pred_root)
    if sigma_by_rel is not None:
        sig = sigma_by_rel.get(os.path.dirname(rel))
        if sig is not None:
            for lim, s, it in sched:
                if sig <= lim:
                    strength, iters = s, it
                    break
    out_path = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pc = np.load(path)
    n_in = pc.shape[0]
    out = pc.astype(np.float64)
    # projection first (fixes the surface), repulsion second (fixes spacing).
    # Projection moves along the normal and repulsion along the tangent, so the
    # order matters little, but this way repulsion has the cleaner surface.
    out = jet_project(out, k=project_k, strength=project_strength,
                      degree=project_degree)
    out = tangential_repulsion(out, k=k, strength=strength, iters=iters)

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
    parser.add_argument('--project_strength', type=float, default=0.0,
                        help='MLS/jet projection onto a locally fitted patch, applied '
                             'before repulsion. 0 disables it, 1 is full projection. '
                             'Lowers P2S, and lowers CD with it since a point d away '
                             'from the surface is at least d from every GT point.')
    parser.add_argument('--project_degree', type=int, default=2, choices=[1, 2],
                        help='2 fits a quadratic and follows curvature; 1 fits a plane '
                             'and flattens more')
    parser.add_argument('--project_k', type=int, default=16,
                        help='neighbours used for the patch fit')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--tau_csv', type=str, default=None,
                        help='per-cloud sigma estimates from predict_on_starter.py '
                             '--save_tau. When given, strength/iters are chosen '
                             'per cloud from --adaptive_sched instead of the global '
                             'values: quiet clouds sit at the projection bound where '
                             'extra pushing only hurts, loud clouds carry tangential '
                             'scramble worth pushing hard against.')
    parser.add_argument('--adaptive_sched', type=str,
                        default='0.010:0.1:1,0.015:0.5:4,999:0.6:5',
                        help='comma list of sigma_limit:strength:iters bands, first '
                             'matching band wins. Default was calibrated on a '
                             'simulation against the organizers\' exact CD metric; '
                             're-sweep on your tune split before trusting it.')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.pred_root, '**', args.pred_filename),
                             recursive=True))
    if not files:
        raise SystemExit(f'no {args.pred_filename} under {args.pred_root}')

    sigma_by_rel, sched = None, None
    if args.tau_csv:
        sigma_by_rel = {}
        with open(args.tau_csv) as f:
            next(f)
            for line in f:
                rel, _, sig = line.strip().rsplit(',', 2)
                sigma_by_rel[rel] = float(sig)
        sched = parse_sched(args.adaptive_sched)
        print(f'adaptive repulsion from {args.tau_csv} '
              f'({len(sigma_by_rel)} clouds), bands {sched}')

    print(f'{len(files)} clouds  |  project(strength={args.project_strength} '
          f'deg={args.project_degree} k={args.project_k})  '
          f'repulse(strength={args.strength} iters={args.iters} k={args.k})')

    fn = partial(process_one, pred_root=args.pred_root, out_root=args.out_root,
                 k=args.k, strength=args.strength, iters=args.iters,
                 project_strength=args.project_strength,
                 project_degree=args.project_degree, project_k=args.project_k,
                 sigma_by_rel=sigma_by_rel, sched=sched)

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
