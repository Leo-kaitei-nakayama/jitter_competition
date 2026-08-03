"""
Is the denoised output actually clumped?

CD trails P2S by a wide margin (62.56 vs 91.32 at the time of writing). One
explanation is that points reach the surface correctly but bunch up on it:
P2S only asks how far each point is from the surface, while CD's reverse term
punishes gaps where a ground-truth point has no prediction nearby.

That is a hypothesis, not a measurement. This script tests it directly by
comparing the nearest-neighbour spacing distribution of the prediction against
the ground truth, which is a uniform sampling of the same surface. If the
prediction is clumped its spacings will be smaller and more spread out, and a
repulsion filter has something to fix. If the two distributions line up, the
CD gap is caused by something else and a spacing filter is wasted effort.

Reported per sample and in aggregate:
    nn_mean       mean distance to the nearest other point
    nn_cv         std/mean of that distance -- the uniformity measure.
                  A blue-noise / Poisson-disk sampling sits low; clumping
                  raises it because tight pairs and empty gaps coexist.
    close_frac    fraction of points whose nearest neighbour is closer than
                  half the ground truth's mean spacing -- direct evidence of
                  points collapsing onto each other.
    cover_p95     95th percentile of, for each GT point, the distance to the
                  nearest predicted point. This is the CD reverse term in
                  distance units: large values mean uncovered regions.

Usage:
    python check_spacing.py --pred_dir ./eval_predict/ss1.5 --gt_dir ./eval_gt
"""
import argparse

import numpy as np
from scipy.spatial import cKDTree

from evaluate import find_samples, normalize_to_unit_sphere


def spacing_stats(pc):
    """Nearest-neighbour distance stats for a point cloud."""
    d, _ = cKDTree(pc).query(pc, k=2)
    nn = d[:, 1]
    return nn.mean(), nn.std() / max(nn.mean(), 1e-12), nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_dir', default='./eval_gt')
    parser.add_argument('--pred_filename', default='denoised.npy')
    parser.add_argument('--gt_filename', default='clean.npy')
    parser.add_argument('--noisy_dir', default='./eval_noisy')
    parser.add_argument('--noisy_filename', default='noisy.npy')
    args = parser.parse_args()

    pred = find_samples(args.pred_dir, args.pred_filename)
    gt = find_samples(args.gt_dir, args.gt_filename)
    noisy = find_samples(args.noisy_dir, args.noisy_filename)
    keys = sorted(set(pred) & set(gt))
    if not keys:
        raise SystemExit('no matching samples -- check --pred_dir / --gt_dir')

    rows = []
    for k in keys:
        g = np.load(gt[k]).astype(np.float64)
        p = np.load(pred[k]).astype(np.float64)

        # normalize both by the GT's bbox, exactly as evaluate.py scores them
        g_n, center, scale = normalize_to_unit_sphere(g)
        p_n = (p - center) / scale

        g_mean, g_cv, _ = spacing_stats(g_n)
        p_mean, p_cv, p_nn = spacing_stats(p_n)

        close_frac = float((p_nn < 0.5 * g_mean).mean())

        # CD reverse term in distance units: how far is each GT point from the
        # nearest prediction?
        cover, _ = cKDTree(p_n).query(g_n, k=1)

        row = dict(key=k, gt_mean=g_mean, gt_cv=g_cv,
                   pred_mean=p_mean, pred_cv=p_cv,
                   close_frac=close_frac, cover_p95=float(np.percentile(cover, 95)))

        if k in noisy:
            n = np.load(noisy[k]).astype(np.float64)
            n_n = (n - center) / scale
            _, n_cv, _ = spacing_stats(n_n)
            row['noisy_cv'] = n_cv
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)

    print('\n' + '=' * 78)
    print('Nearest-neighbour spacing: prediction vs ground truth')
    print('=' * 78)
    cols = ['gt_mean', 'pred_mean', 'gt_cv', 'pred_cv', 'close_frac', 'cover_p95']
    if 'noisy_cv' in df:
        cols.insert(4, 'noisy_cv')
    print(df[cols].agg(['mean', 'std', 'min', 'max']).to_string())

    gt_cv, pred_cv = df['gt_cv'].mean(), df['pred_cv'].mean()
    ratio = df['pred_mean'].mean() / df['gt_mean'].mean()

    print('\n' + '=' * 78)
    print('Verdict')
    print('=' * 78)
    print(f'  spacing uniformity (cv):   GT {gt_cv:.3f}   pred {pred_cv:.3f}   '
          f'({pred_cv / gt_cv:.2f}x)')
    print(f'  mean spacing ratio pred/GT: {ratio:.3f}')
    print(f'  points with a neighbour closer than half GT spacing: '
          f'{df["close_frac"].mean():.1%}')

    if pred_cv > 1.25 * gt_cv or df['close_frac'].mean() > 0.10:
        print('\n  -> CLUMPED. The prediction is measurably less uniform than the')
        print('     ground truth. A tangential repulsion filter should reduce CD')
        print('     without disturbing P2S.')
    elif pred_cv < 1.05 * gt_cv:
        print('\n  -> NOT CLUMPED. Spacing matches the ground truth closely, so the')
        print('     CD gap is caused by something else and a repulsion filter will')
        print('     not help. Do not build it.')
    else:
        print('\n  -> MARGINAL. Some excess non-uniformity but not dramatic. A')
        print('     repulsion filter would likely give a small gain at best.')

    df.to_csv('spacing_stats.csv', index=False)
    print('\nSaved spacing_stats.csv (per-sample detail)')


if __name__ == '__main__':
    main()
