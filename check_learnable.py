"""
Is the frozen model's leftover error learnable?

A second stage on top of a frozen backbone can only help if the residual error
is predictable from what that stage can see. Two possibilities:

  systematic   the error is a consistent function of local geometry -- the
               model always pulls edges in, always under-moves in sparse
               regions. A trained layer can learn that and undo it.
  random       the error is estimation noise with no structure. Nothing can
               recover it, and any filter that tries will make things worse.
               This is exactly why jet projection failed.

Two tests, both computed only from information a stage-2 network would have
(the denoised cloud), against the true residual (clean - denoised, exact here
because noisy.npy is clean.npy plus a displacement):

  1. SPATIAL AUTOCORRELATION
     Does a point's error resemble its neighbours' errors? A smooth error
     field is predictable from local context; an uncorrelated one is not.
     Reported as the correlation between each point's normal-direction error
     and the mean of its k neighbours'. Near 0 means nothing to learn.

  2. LINEAR PROBE
     Ridge regression from cheap local descriptors -- planarity, curvature,
     density, and the surface normal -- onto the normal-direction error, scored
     out-of-sample. A linear probe is far weaker than a network, so a small
     positive R^2 still implies real signal, while an R^2 at or below zero says
     the obvious geometric cues carry none.

Usage:
    python check_learnable.py --pred_dir ./eval_predict/ens06 --gt_dir ./eval_gt_tune
"""
import argparse

import numpy as np
from scipy.spatial import cKDTree

from evaluate import find_samples, normalize_to_unit_sphere


def local_features(pc, k=16):
    """Descriptors a stage-2 network could compute from the denoised cloud."""
    d, idx = cKDTree(pc).query(pc, k=k + 1)
    patch = pc[idx]
    cen = patch - patch.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', cen, cen) / (k + 1)
    evals, evecs = np.linalg.eigh(cov)               # ascending
    normals = evecs[:, :, 0]

    l0, l1, l2 = evals[:, 0], evals[:, 1], evals[:, 2]
    tot = np.maximum(l0 + l1 + l2, 1e-20)
    feats = np.column_stack([
        l0 / tot,                       # curvature / surface variation
        (l1 - l0) / tot,                # planarity
        (l2 - l1) / tot,                # linearity
        d[:, 1],                        # nearest-neighbour spacing
        d[:, 1:].mean(axis=1),          # mean neighbourhood spacing
        np.abs(normals),                # normal orientation, sign-free
    ])
    return feats, normals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_dir', default='./eval_gt_tune')
    parser.add_argument('--pred_filename', default='denoised.npy')
    parser.add_argument('--gt_filename', default='clean.npy')
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--max_clouds', type=int, default=20)
    parser.add_argument('--points_per_cloud', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    pred = find_samples(args.pred_dir, args.pred_filename)
    gt = find_samples(args.gt_dir, args.gt_filename)
    keys = sorted(set(pred) & set(gt))[:args.max_clouds]
    if not keys:
        raise SystemExit('no matching samples')

    rng = np.random.RandomState(args.seed)
    X_all, y_all, autocorr = [], [], []

    for key in keys:
        g = np.load(gt[key]).astype(np.float64)
        p = np.load(pred[key]).astype(np.float64)
        if g.shape != p.shape:
            continue

        g_n, center, scale = normalize_to_unit_sphere(g)
        p_n = (p - center) / scale

        feats, normals = local_features(p_n, k=args.k)

        # Residual toward the NEAREST clean point, not the same-index one.
        #
        # The model moves points along the surface, so output point i is near
        # the surface but generally not near clean point i -- that tangential
        # scrambling is several point spacings wide and is arbitrary, since
        # which clean point a given output point drifted toward carries no
        # information. Using g_n[i] - p_n[i] measures mostly that scrambling,
        # which leaks into the normal component through imperfect normals and
        # swamps the quantity of interest.
        #
        # A stage-2 network would reduce distance to the SURFACE, so that is
        # what the target has to be. Sanity check: the RMS printed below should
        # land near sqrt(P2S_pred) ~ 0.004. An order of magnitude more means
        # tangential displacement is contaminating it again.
        _, nn_j = cKDTree(g_n).query(p_n, k=1)
        resid = g_n[nn_j] - p_n
        rn = np.einsum('ni,ni->n', resid, normals)

        # 1. does a point's error look like its neighbours' errors?
        _, idx = cKDTree(p_n).query(p_n, k=args.k + 1)
        nbr_mean = rn[idx[:, 1:]].mean(axis=1)
        if rn.std() > 1e-15 and nbr_mean.std() > 1e-15:
            autocorr.append(float(np.corrcoef(rn, nbr_mean)[0, 1]))

        sel = rng.choice(len(rn), size=min(args.points_per_cloud, len(rn)),
                         replace=False)
        X_all.append(feats[sel])
        y_all.append(rn[sel])

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    # standardise, then ridge with a held-out half
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
    y_c = y - y.mean()
    n = len(y)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]

    A = X[tr].T @ X[tr] + 1e-3 * len(tr) * np.eye(X.shape[1])
    w = np.linalg.solve(A, X[tr].T @ y_c[tr])
    pred_te = X[te] @ w
    ss_res = float(((y_c[te] - pred_te) ** 2).sum())
    ss_tot = float(((y_c[te] - y_c[te].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    ac = float(np.mean(autocorr)) if autocorr else float('nan')

    print('\n' + '=' * 70)
    print(f'Residual error structure  ({len(keys)} clouds, {n} points)')
    print('=' * 70)
    print(f'  normal-direction error, RMS:      {y.std():.6f}'
          f'   (expect ~0.004; much larger means the target is contaminated)')
    print(f'  spatial autocorrelation:          {ac:+.3f}')
    print(f'  linear probe R^2 (out of sample): {r2:+.4f}')

    print('\n' + '=' * 70)
    print('Verdict')
    print('=' * 70)
    if ac > 0.35 or r2 > 0.05:
        print('  LEARNABLE. The error has structure a stage-2 network could pick up.')
        print('  A linear probe is far weaker than a network, so this is a floor,')
        print('  not a ceiling. Worth building.')
    elif ac < 0.15 and r2 < 0.01:
        print('  NOT LEARNABLE from local geometry. The residual behaves like')
        print('  estimation noise, so a refinement stage reading the denoised cloud')
        print('  has nothing to work with -- the same reason jet projection failed.')
        print('  If you build stage 2 anyway, feed it the frozen backbone FEATURES')
        print('  rather than the output coordinates; those carry information the')
        print('  coordinates have already thrown away.')
    else:
        print('  MARGINAL. Some structure, but weak. A stage-2 network reading only')
        print('  geometry would likely gain little; the backbone features are the')
        print('  more promising input.')


if __name__ == '__main__':
    main()
