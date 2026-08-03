"""
Visualize noisy / predicted / GT point clouds side by side, with per-point
error coloring, as an interactive HTML file (Plotly -- rotate/zoom in browser).

For each sample it shows 3 panels:
  1. Noisy input,      colored by distance to GT (how bad the noise was)
  2. Model prediction,  colored by distance to GT (how bad the residual error is)
  3. Clean GT,          plain color (reference)

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
        rows=n_rows, cols=3,
        specs=[[{'type': 'scene'}] * 3 for _ in range(n_rows)],
        subplot_titles=sum(
            ([f'{k}<br>Noisy (colored by err-to-GT)',
              f'{k}<br>Predicted (colored by err-to-GT)',
              f'{k}<br>GT'] for k in keys), []
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

        # Per-point error: distance to nearest GT point
        err_noisy = nn_dist(noisy, gt)
        err_pred = nn_dist(pred, gt)

        # Shared color scale so noisy vs pred panels are visually comparable
        cmax = float(np.percentile(np.concatenate([err_noisy, err_pred]), 98))
        cmin = 0.0

        gt_s, = subsample(gt, max_points=args.max_points)
        noisy_s, err_noisy_s = subsample(noisy, err_noisy, max_points=args.max_points)
        pred_s, err_pred_s = subsample(pred, err_pred, max_points=args.max_points)

        fig.add_trace(make_scatter(noisy_s, err_noisy_s, 'Reds', 'noisy',
                                    showscale=(row == 1), cmin=cmin, cmax=cmax),
                       row=row, col=1)
        fig.add_trace(make_scatter(pred_s, err_pred_s, 'Reds', 'pred',
                                    showscale=False, cmin=cmin, cmax=cmax),
                       row=row, col=2)
        fig.add_trace(make_scatter(gt_s, '#2ca02c', None, 'gt'),
                       row=row, col=3)

        print(f'{key}: mean err noisy={err_noisy.mean():.5f}  mean err pred={err_pred.mean():.5f}  '
              f'(cmax used for color scale={cmax:.5f})')

    fig.update_layout(
        height=520 * n_rows,
        width=1500,
        title='Noisy vs Predicted vs GT -- point color = distance to nearest GT point',
        showlegend=False,
    )
    # Lock equal aspect ratio per scene so shapes aren't visually distorted
    for i in range(1, n_rows * 3 + 1):
        scene_key = 'scene' if i == 1 else f'scene{i}'
        fig.update_layout(**{scene_key: dict(aspectmode='data')})

    fig.write_html(args.out_html)
    print(f'\nSaved interactive visualization to: {args.out_html}')


if __name__ == '__main__':
    main()