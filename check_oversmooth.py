"""
Is the model destroying real geometry along with the noise?

Oversmoothing has a specific signature: curvature disappears. Noise inflates
apparent curvature, the clean surface has its own true curvature, and a
denoiser that overshoots flattens the result BELOW the clean value. So the
three-way ordering tells the story directly:

    curv(noisy) > curv(clean) ~ curv(pred)     healthy
    curv(noisy) > curv(clean) > curv(pred)     oversmoothing

Curvature here is Pauly et al.'s surface variation, lambda0 / (lambda0 +
lambda1 + lambda2) over a k-neighbourhood's covariance -- the fraction of local
spread that leaves the best-fit plane. It needs no mesh, no normals, and no
orientation.

The per-bin table is the actionable part: it splits the clean points by their
OWN curvature and reports what the model did in each band. A model that only
loses detail where detail exists shows a ratio that falls as curvature rises.
Flat regions are the control -- if the ratio drops there too, the cause is
noise-level miscalibration rather than detail loss.

Usage:
    python check_oversmooth.py \
        --pred_root results/tune_base \
        --clean_root eval_gt_tune \
        --noisy_root eval_noisy_tune
"""
import os
import argparse

import numpy as np
from scipy.spatial import cKDTree


def surface_variation(pc, k=16):
    """Pauly's surface variation per point: lambda0 / sum(lambda). (N,)"""
    _, idx = cKDTree(pc).query(pc, k=k + 1)
    patch = pc[idx]                                     # (N, k+1, 3)
    cen = patch - patch.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', cen, cen) / (k + 1)
    ev = np.linalg.eigvalsh(cov)                        # ascending
    return ev[:, 0] / np.maximum(ev.sum(axis=1), 1e-20)


def find_keys(root, name):
    keys = []
    for dirpath, _, files in os.walk(root):
        if name in files:
            keys.append(os.path.relpath(dirpath, root))
    return sorted(keys)


def main(args):
    keys = find_keys(args.clean_root, args.clean_name)[:args.n_clouds]
    if not keys:
        raise SystemExit(f'no {args.clean_name} under {args.clean_root}')

    edges = np.array(args.bins, dtype=np.float64)
    n_bin = len(edges) - 1
    agg = {k: np.zeros(n_bin) for k in ('n', 'c_clean', 'c_pred', 'dist')}
    tot = {'noisy': [], 'clean': [], 'pred': []}

    for key in keys:
        clean = np.load(os.path.join(args.clean_root, key, args.clean_name)).astype(np.float64)
        pred_p = os.path.join(args.pred_root, key, args.pred_name)
        if not os.path.exists(pred_p):
            continue
        pred = np.load(pred_p).astype(np.float64)

        cv_clean = surface_variation(clean, args.k)
        cv_pred = surface_variation(pred, args.k)
        tot['clean'].append(cv_clean)
        tot['pred'].append(cv_pred)

        if args.noisy_root:
            noisy_p = os.path.join(args.noisy_root, key, args.noisy_name)
            if os.path.exists(noisy_p):
                tot['noisy'].append(
                    surface_variation(np.load(noisy_p).astype(np.float64), args.k))

        # pair each clean point with its nearest predicted point
        d, j = cKDTree(pred).query(clean, k=1)
        cvp = cv_pred[j]
        b = np.digitize(cv_clean, edges) - 1
        for i in range(n_bin):
            m = b == i
            if not m.any():
                continue
            agg['n'][i] += m.sum()
            agg['c_clean'][i] += cv_clean[m].sum()
            agg['c_pred'][i] += cvp[m].sum()
            agg['dist'][i] += d[m].sum()

    def cat(name):
        return np.concatenate(tot[name]) if tot[name] else None

    print(f'{len(keys)} clouds, k={args.k} neighbourhood\n')
    print('=== overall surface variation (higher = more curved) ===')
    cl, pr, no = cat('clean'), cat('pred'), cat('noisy')
    if no is not None:
        print(f'  noisy input : {no.mean():.5f}   (noise inflates this)')
    print(f'  clean GT    : {cl.mean():.5f}   <- the target')
    print(f'  prediction  : {pr.mean():.5f}')
    r = pr.mean() / cl.mean()
    print(f'\n  pred / clean = {r:.3f}', end='  ')
    if r < 0.85:
        print('=> OVERSMOOTHING: geometry is being flattened')
    elif r > 1.15:
        print('=> UNDER-denoised: residual noise still inflating curvature')
    else:
        print('=> healthy: curvature preserved')

    print('\n=== by the clean surface\'s own curvature ===')
    print('curvature band        n      curv(clean)  curv(pred)  ratio   mean dist')
    for i in range(n_bin):
        if agg['n'][i] == 0:
            continue
        n = agg['n'][i]
        cc, cp, dd = agg['c_clean'][i] / n, agg['c_pred'][i] / n, agg['dist'][i] / n
        lbl = f'{edges[i]:.4f}-{edges[i+1]:.4f}'
        print(f'{lbl:18s} {int(n):8d}   {cc:.5f}     {cp:.5f}    {cp/cc:5.2f}   {dd:.6f}')
    print('\nA ratio that falls as curvature rises = detail is being lost where')
    print('detail exists. A ratio flat and below 1 everywhere = the model is')
    print('over-denoising uniformly, which points at the noise-level estimate.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pred_root', type=str, required=True)
    p.add_argument('--clean_root', type=str, required=True)
    p.add_argument('--noisy_root', type=str, default='')
    p.add_argument('--pred_name', type=str, default='denoised.npy')
    p.add_argument('--clean_name', type=str, default='clean.npy')
    p.add_argument('--noisy_name', type=str, default='noisy.npy')
    p.add_argument('--n_clouds', type=int, default=10)
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--bins', type=float, nargs='+',
                   default=[0.0, 0.002, 0.005, 0.010, 0.020, 1.0])
    args = p.parse_args()
    main(args)
