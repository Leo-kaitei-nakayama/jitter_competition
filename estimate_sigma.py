"""
Model-free per-cloud noise estimation, for driving the adaptive repulsion
filter without a 36-minute re-predict.

Method: fit a local quadric patch (same math as postprocess.jet_project)
around a subsample of points and read the noise off the robust spread (MAD)
of point-to-patch residuals. That raw spread is NOT proportional to sigma --
the point takes part in its own fit, the normal component of Laplace noise
needs a non-Gaussian MAD factor, and at high sigma the neighbourhood mixes
across the noise thickness and the fit saturates -- so instead of a constant
we invert an empirical raw->sigma curve, measured on synthetic
sphere/ellipsoid/cube clouds at 50k points in the competition's normalized
frame. Across those surfaces the raw value agrees within 4-7% for sigma
<= 0.0075 and 13-18% through 0.015, which is what band selection needs.

Two caveats. Above sigma ~0.03 the curve flattens, so estimates there are
lower bounds ("loud") rather than measurements. And the anchors bake in the
~50k-point sampling density: for round B clouds with a very different point
count, re-run the calibration in the commit message and refresh the tables.

Output CSV is drop-in compatible with postprocess.py --tau_csv. The tau
column is -1: this estimator never touches the network.

Usage:
    python estimate_sigma.py --noisy_root ./eval_noisy_tune --out_csv taus_tune.csv
    python postprocess.py ... --tau_csv taus_tune.csv
"""
import os
import argparse

import numpy as np
from scipy.spatial import cKDTree

# raw MAD-scale residual -> sigma, measured on synthetic sphere/ellipsoid/cube
# at N=50k in the normalized frame (mean over the three surfaces)
RAW_ANCHORS = [0.00234, 0.00391, 0.00575, 0.00733, 0.00865,
               0.00956, 0.01088, 0.01137, 0.01202]
SIGMA_ANCHORS = [0.003, 0.005, 0.0075, 0.010, 0.0125,
                 0.015, 0.020, 0.025, 0.030]


def normalize_unit_sphere(pc):
    p_max = pc.max(axis=0); p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc ** 2).sum(axis=1).max())
    return pc / scale


def quadric_residuals(pc, query_idx, k=16, ridge=1e-8):
    """Signed distance of each query point to the quadric fitted through its
    k+1 neighbourhood (self included). Math identical to jet_project."""
    _, idx = cKDTree(pc).query(pc[query_idx], k=k + 1)
    patch = pc[idx]
    centroid = patch.mean(axis=1, keepdims=True)
    cen = patch - centroid

    cov = np.einsum('nki,nkj->nij', cen, cen) / (k + 1)
    _, vecs = np.linalg.eigh(cov)
    n_ax, t1, t2 = vecs[:, :, 0], vecs[:, :, 1], vecs[:, :, 2]

    a = np.einsum('nki,ni->nk', cen, t1)
    b = np.einsum('nki,ni->nk', cen, t2)
    c = np.einsum('nki,ni->nk', cen, n_ax)
    A = np.stack([np.ones_like(a), a, b, a * a, a * b, b * b], axis=-1)
    m = A.shape[-1]
    AtA = np.einsum('nkm,nkl->nml', A, A) + ridge * np.eye(m)
    Atc = np.einsum('nkm,nk->nm', A, c)
    coef = np.linalg.solve(AtA, Atc[..., None])[..., 0]

    self_rel = pc[query_idx] - centroid[:, 0, :]
    sa = np.einsum('ni,ni->n', self_rel, t1)
    sb = np.einsum('ni,ni->n', self_rel, t2)
    sc = np.einsum('ni,ni->n', self_rel, n_ax)
    basis = np.stack([np.ones_like(sa), sa, sb, sa * sa, sa * sb, sb * sb], axis=-1)
    fitted = np.einsum('nm,nm->n', coef, basis)
    return fitted - sc


def estimate_sigma(pc, k=16, samples=8000, seed=0):
    pc = normalize_unit_sphere(pc.astype(np.float64))
    rng = np.random.default_rng(seed)
    q = rng.choice(pc.shape[0], size=min(samples, pc.shape[0]), replace=False)
    r = quadric_residuals(pc, q, k=k)
    raw = 1.4826 * np.median(np.abs(r - np.median(r)))
    return float(np.interp(raw, RAW_ANCHORS, SIGMA_ANCHORS))


def main(args):
    keys = []
    for dirpath, _, files in os.walk(args.noisy_root):
        if args.data_name in files:
            keys.append(os.path.relpath(dirpath, args.noisy_root))
    if not keys:
        raise SystemExit(f'no {args.data_name} under {args.noisy_root}')

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    with open(args.out_csv, 'w') as f:
        f.write('rel,tau,sigma_est\n')
        for key in sorted(keys):
            pc = np.load(os.path.join(args.noisy_root, key, args.data_name))
            s = estimate_sigma(pc, k=args.k, samples=args.samples, seed=args.seed)
            f.write(f'{key},-1,{s:.6f}\n')
            print(f'{key}: sigma_est = {s:.4f}')
    print(f'wrote {len(keys)} estimates to {args.out_csv}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--noisy_root', type=str, required=True)
    p.add_argument('--data_name', type=str, default='noisy.npy')
    p.add_argument('--out_csv', type=str, required=True)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--samples', type=int, default=8000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    main(args)
