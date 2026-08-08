"""
Visualize noisy / predicted / GT point clouds side by side, with per-point
error coloring, as an interactive HTML file (Plotly -- rotate/zoom in browser).

For each sample it shows 4 panels, one per way of losing score:
  1. Noisy input,      colored by distance to GT (how bad the noise was)
  2. Model prediction, colored by distance to GT -- CD's FIRST term
                       (red = predicted points far from any clean point)
  3. Clean GT,         colored by distance to the nearest PREDICTED point --
                       CD's SECOND term (red = surface the prediction left
                       uncovered: the holes; this is where our CD score dies)
  4. Model prediction, colored by its own nearest-neighbour spacing
                       (dark = points crowded into clumps -- the same failure
                       as panel 3 seen from the other side: every clump
                       somewhere means a hole somewhere else)

Usage:
    python visualize_errors.py \
        --keys 04379243/4afbcdeba648df2e19fb4103277a6b93,04468005/40fcd2ccc96b3fbd041917556492646 \
        --gt_dir ./eval_gt --noisy_dir ./eval_noisy --pred_dir ./eval_predict \
        --gt_filename clean.npy --noisy_filename noisy.npy --pred_filename denoised.npy \
        --max_points 15000 \
        --out_html ./error_viz.html

Then download error_viz.html to your own machine and open it in a browser
(or serve it: `python -m http.server` in that directory and open the link).
"""

import argparse
import os
import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load(path):
    return np.load(path).astype(np.float64)


def nn_dist(a, b):
    """For each point in a, distance to nearest point in b."""
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    return d


def subsample(*arrays, max_points):
    n = arrays[0].shape[0]
    if n <= max_points:
        return arrays
    idx = np.random.choice(n, max_points, replace=False)
    return tuple(a[idx] for a in arrays)


def make_scatter(pts, color, colorscale, name, showscale=False, cmin=None, cmax=None):
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=color,
            colorscale=colorscale,
            showscale=showscale,
            cmin=cmin, cmax=cmax,
            colorbar=dict(title='dist to GT', x=1.0) if showscale else None,
        ),
        name=name,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys', type=str, required=True,
                         help='comma-separated list of "<category>/<model_id>" to visualize')
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--noisy_dir', type=str, required=True)
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--gt_filename', type=str, default='clean.npy')
    parser.add_argument('--noisy_filename', type=str, default='noisy.npy')
    parser.add_argument('--pred_filename', type=str, default='denoised.npy')
    parser.add_argument('--max_points', type=int, default=15000,
                         help='subsample each cloud to this many points for browser performance')
    parser.add_argument('--out_html', type=str, default='./error_viz.html')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    keys = [k.strip() for k in args.keys.split(',') if k.strip()]

    n_rows = len(keys)
    fig = make_subplots(
        rows=n_rows, cols=4,
        specs=[[{'type': 'scene'}] * 4 for _ in range(n_rows)],
        subplot_titles=sum(
            ([f'{k}<br>Noisy (err to GT)',
              f'{k}<br>Pred (err to GT = CD term 1)',
              f'{k}<br>GT (dist to nearest pred = CD term 2: holes)',
              f'{k}<br>Pred (NN spacing: dark = clumps)'] for k in keys), []
        ),
        vertical_spacing=0.06,
    )

    for row, key in enumerate(keys, start=1):
        gt_path = os.path.join(args.gt_dir, key, args.gt_filename)
        noisy_path = os.path.join(args.noisy_dir, key, args.noisy_filename)
        pred_path = os.path.join(args.pred_dir, key, args.pred_filename)

        for p in (gt_path, noisy_path, pred_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f'Missing file for key "{key}": {p}')

        gt = load(gt_path)
        noisy = load(noisy_path)
        pred = load(pred_path)

        # CD term 1: each predicted point's distance to the nearest GT point
        err_noisy = nn_dist(noisy, gt)
        err_pred = nn_dist(pred, gt)
        # CD term 2: each GT point's distance to the nearest predicted point.
        # This is the term the per-point loss cannot see and the one our score
        # actually loses on -- red patches here ARE the missing CD points.
        coverage = nn_dist(gt, pred)
        # Clump map: prediction's own nearest-neighbour spacing. Mass surplus
        # (clumps) and mass deficit (holes) are two views of one failure.
        spacing = cKDTree(pred).query(pred, k=2)[0][:, 1]

        # Shared color scale so noisy vs pred panels are visually comparable
        cmax = float(np.percentile(np.concatenate([err_noisy, err_pred]), 98))
        cmin = 0.0
        cov_max = float(np.percentile(coverage, 98))
        sp_max = float(np.percentile(spacing, 98))

        noisy_s, err_noisy_s = subsample(noisy, err_noisy, max_points=args.max_points)
        pred_s, err_pred_s, spacing_s = subsample(pred, err_pred, spacing,
                                                  max_points=args.max_points)
        gt_s, coverage_s = subsample(gt, coverage, max_points=args.max_points)

        fig.add_trace(make_scatter(noisy_s, err_noisy_s, 'Reds', 'noisy',
                                    showscale=(row == 1), cmin=cmin, cmax=cmax),
                       row=row, col=1)
        fig.add_trace(make_scatter(pred_s, err_pred_s, 'Reds', 'pred',
                                    showscale=False, cmin=cmin, cmax=cmax),
                       row=row, col=2)
        fig.add_trace(make_scatter(gt_s, coverage_s, 'Reds', 'gt coverage',
                                    showscale=False, cmin=0.0, cmax=cov_max),
                       row=row, col=3)
        fig.add_trace(make_scatter(pred_s, spacing_s, 'Viridis', 'pred spacing',
                                    showscale=False, cmin=0.0, cmax=sp_max),
                       row=row, col=4)

        print(f'{key}:')
        print(f'  CD term 1 (pred->GT)  mean {err_pred.mean():.5f}   [noisy was {err_noisy.mean():.5f}]')
        print(f'  CD term 2 (GT->pred)  mean {coverage.mean():.5f}   <- the holes')
        print(f'  pred NN spacing: median {np.median(spacing):.5f}, '
              f'p5 {np.percentile(spacing, 5):.5f} (clumps), '
              f'p95 {np.percentile(spacing, 95):.5f} (gaps)')

    fig.update_layout(
        height=520 * n_rows,
        width=2000,
        title='Noisy | Pred err (CD term 1) | GT coverage (CD term 2: holes) | Pred spacing (clumps)',
        showlegend=False,
    )
    # Lock equal aspect ratio per scene so shapes aren't visually distorted
    for i in range(1, n_rows * 4 + 1):
        scene_key = 'scene' if i == 1 else f'scene{i}'
        fig.update_layout(**{scene_key: dict(aspectmode='data')})

    fig.write_html(args.out_html)
    print(f'\nSaved interactive visualization to: {args.out_html}')


if __name__ == '__main__':
    main()