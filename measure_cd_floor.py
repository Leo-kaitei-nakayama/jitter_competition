"""
Measure how much of the CD score is reachable at all.

CD compares the prediction against clean.npy, which is one particular random
sampling of the mesh surface. Any other sampling of that same surface -- even a
geometrically perfect answer -- sits some distance away, purely because the two
point sets do not line up. That distance is a floor on CD_pred.

The score is 100 * (1 - CD_pred / CD_noisy). On a noisy cloud CD_noisy is large
and the floor is negligible. On a clean cloud CD_noisy can be the same order as
the floor, and then the score is capped no matter how good the denoiser is.

This script estimates the floor by resampling each eval mesh with a different
seed and measuring CD against clean.npy, then reports the best cd_score any
method could possibly achieve on each sample.

Usage:
    python measure_cd_floor.py \
        --gt_dir ./eval_gt --noisy_dir ./eval_noisy --mesh_dir ./eval_mesh_normalized \
        --meta_csv ./eval_meta.csv --scores_csv ./scores_ep049.csv
"""
import argparse
import os

import numpy as np
import trimesh

from evaluate import chamfer_distance, find_samples, find_meshes


def sample_mesh(path, n, seed):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    rng = np.random.RandomState(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=rng.randint(1 << 30))
    return np.asarray(pts, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default='./eval_gt')
    parser.add_argument('--noisy_dir', default='./eval_noisy')
    parser.add_argument('--mesh_dir', default='./eval_mesh_normalized')
    parser.add_argument('--mesh_data_name', default='models/model_normalized.obj')
    parser.add_argument('--meta_csv', default='./eval_meta.csv')
    parser.add_argument('--scores_csv', default='',
                        help='optional: evaluate.py --save_csv output, to compare '
                             'the achieved cd_score against the ceiling')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    gt = find_samples(args.gt_dir, 'clean.npy')
    noisy = find_samples(args.noisy_dir, 'noisy.npy')
    meshes = find_meshes(args.mesh_dir, args.mesh_data_name)

    keys = sorted(set(gt) & set(noisy) & set(meshes))
    if not keys:
        raise SystemExit('no samples found -- check the three --*_dir paths')

    import pandas as pd
    meta = None
    if args.meta_csv and os.path.exists(args.meta_csv):
        meta = pd.read_csv(args.meta_csv, dtype={'category': str, 'model_id': str})
        meta = dict(zip(meta['key'], meta['noise_std']))

    achieved = None
    if args.scores_csv and os.path.exists(args.scores_csv):
        sc = pd.read_csv(args.scores_csv, dtype={'category': str, 'model_id': str})
        achieved = dict(zip(sc['key'], sc['cd_score']))

    rows = []
    for k in keys:
        clean = np.load(gt[k]).astype(np.float64)
        noi = np.load(noisy[k]).astype(np.float64)
        resampled = sample_mesh(meshes[k], clean.shape[0], args.seed)

        cd_floor = chamfer_distance(resampled, clean)   # perfect-answer CD
        cd_noisy = chamfer_distance(noi, clean)         # scoring baseline
        ceiling = max(0.0, min(100.0, 100.0 * (1.0 - cd_floor / cd_noisy)))

        rows.append({
            'key': k,
            'noise_std': meta.get(k, np.nan) if meta else np.nan,
            'cd_floor': cd_floor,
            'cd_noisy': cd_noisy,
            'cd_score_ceiling': ceiling,
            'cd_score_achieved': achieved.get(k, np.nan) if achieved else np.nan,
        })
        print(f'  {k}  ceiling={ceiling:6.2f}'
              + (f'  achieved={rows[-1]["cd_score_achieved"]:6.2f}' if achieved else ''))

    df = pd.DataFrame(rows)
    df['headroom'] = df['cd_score_ceiling'] - df['cd_score_achieved']

    print('\n' + '=' * 70)
    print('Best CD score any method could reach on this eval set')
    print('=' * 70)
    cols = ['cd_score_ceiling'] + (['cd_score_achieved', 'headroom'] if achieved else [])
    print(df[cols].agg(['mean', 'std', 'min', 'max']).to_string())

    if df['noise_std'].notna().any():
        print('\n' + '=' * 70)
        print('By noise level (quartile bins)')
        print('=' * 70)
        df['noise_bin'] = pd.qcut(df['noise_std'], q=4, duplicates='drop')
        print(df.groupby('noise_bin', observed=True)[cols].mean().to_string())
        print('\nIf the ceiling drops on the low-noise bins as sharply as the')
        print('achieved score does, the low-noise CD penalty is a property of the')
        print('metric, not of the model, and no amount of denoising will fix it.')

    df.to_csv('cd_floor.csv', index=False)
    print('\nSaved cd_floor.csv')


if __name__ == '__main__':
    main()
