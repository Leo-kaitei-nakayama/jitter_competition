"""
How often is Eq. 14's nearest-neighbour target wrong, and by how much?

compute_gt_score approximates the training target as S(x) = NN(x, x_clean) - x.
bridge/data_bridge.py cuts the noisy and clean patches with the same indices,
so the exact target is clean[i] - noisy[i] -- no search needed.

The two agree only when a displaced point's nearest clean neighbour is still its
own origin. That fails once the displacement grows relative to the point
spacing, and when it fails the target points sideways, toward whichever
neighbour happens to be closest -- which biases predictions toward locally dense
regions and clusters points together.

This script measures the disagreement directly on real training meshes, with no
GPU and no training run, so you can tell in advance whether --exact_score is
worth a retrain. If the targets barely differ, the flag will not change
anything. If they diverge sharply at the noise levels you care about, it will.

Usage:
    python check_score_target.py --root ./dataset_train --datalist ./datalist/train.txt
"""
import argparse
import math

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def normalize_unit_sphere(pc):
    p_max, p_min = pc.max(axis=0), pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    return (pc / np.sqrt((pc ** 2).sum(axis=1).max())).astype(np.float32)


def sample_mesh(path, n, rng):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=rng.randint(1 << 30))
    return np.asarray(pts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./dataset_train')
    parser.add_argument('--datalist', default='./datalist/train.txt')
    parser.add_argument('--mesh_name', default='models/model_normalized.obj')
    parser.add_argument('--num_shapes', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=32768)
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--noise_dist', default='laplace', choices=['laplace', 'gaussian'])
    parser.add_argument('--sigmas', type=float, nargs='+',
                        default=[0.005, 0.0075, 0.010, 0.015, 0.020])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    with open(args.datalist) as f:
        entries = [l.strip() for l in f if l.strip()]
    chosen = rng.choice(entries, size=min(args.num_shapes, len(entries)), replace=False)

    import os
    stats = {s: [] for s in args.sigmas}
    spacing_all = []

    for rel in chosen:
        path = os.path.join(args.root, rel, args.mesh_name)
        if not os.path.exists(path):
            continue
        clean_full = normalize_unit_sphere(sample_mesh(path, args.num_samples, rng))

        for sigma in args.sigmas:
            if args.noise_dist == 'gaussian':
                noise = rng.normal(0, sigma, size=clean_full.shape).astype(np.float32)
            else:
                noise = rng.laplace(0, sigma / math.sqrt(2.0),
                                    size=clean_full.shape).astype(np.float32)
            noisy_full = clean_full + noise

            # same patch construction as data_bridge
            seed_i = rng.randint(clean_full.shape[0])
            _, nn_idx = cKDTree(noisy_full).query(noisy_full[seed_i][None, :],
                                                  k=args.patch_size)
            nn_idx = nn_idx[0]
            noisy = noisy_full[nn_idx]
            clean = clean_full[nn_idx]     # row i of one IS row i of the other

            # mean spacing of the clean patch, for context
            d, _ = cKDTree(clean).query(clean, k=2)
            spacing_all.append(float(d[:, 1].mean()))

            # Eq. 14: nearest clean neighbour within the patch
            nn_d, nn_j = cKDTree(clean).query(noisy, k=1)

            exact_t = clean - noisy               # (M,3) exact displacement
            nn_t = clean[nn_j] - noisy            # (M,3) Eq. 14 target

            wrong = (nn_j != np.arange(len(nn_j)))          # picked another point
            diff = np.linalg.norm(exact_t - nn_t, axis=1)   # how far apart the targets are
            mag_e = np.linalg.norm(exact_t, axis=1)

            stats[sigma].append((
                wrong.mean(),
                float(diff.mean() / max(mag_e.mean(), 1e-9)),
                float(np.linalg.norm(nn_t, axis=1).mean() / max(mag_e.mean(), 1e-9)),
            ))

    print(f'\nmean clean-patch point spacing: {np.mean(spacing_all):.5f}')
    print('\n' + '=' * 72)
    print('Disagreement between Eq. 14 (NN) and the exact target')
    print('=' * 72)
    print(f'{"sigma":>8} {"sigma/spacing":>14} {"NN picks other pt":>19} '
          f'{"|diff|/|exact|":>16} {"|NN|/|exact|":>14}')
    for s in args.sigmas:
        if not stats[s]:
            continue
        a = np.array(stats[s])
        print(f'{s:>8.4f} {s / np.mean(spacing_all):>14.2f} {a[:, 0].mean():>18.1%} '
              f'{a[:, 1].mean():>16.3f} {a[:, 2].mean():>14.3f}')

    print("""
Reading this:
  NN picks other pt  fraction of points whose nearest clean neighbour is NOT
                     their own origin. Where this is near 0 the two targets are
                     identical and --exact_score cannot change anything.
  |diff|/|exact|     size of the disagreement relative to the true displacement.
                     Above ~0.3 the targets are teaching materially different
                     things.
  |NN|/|exact|       below 1 means Eq. 14 systematically UNDER-states how far
                     points must move -- consistent with the model
                     under-denoising, which is why --sigma_scale 1.5 helped.
""")


if __name__ == '__main__':
    main()
