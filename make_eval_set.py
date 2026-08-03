"""
Build a local evaluation set: pick N ShapeNet models from the training
datalist, sample clean point clouds from their meshes, add noise the same
way the competition's test set was generated, and save clean/noisy/mesh
triples that `evaluate.py` can score against.

This lets you measure CD/P2S scores locally, without using up your
2-submissions-per-day limit on the real (GT-less) test set.

Usage:
    python make_eval_set.py \
        --root ./dataset_train \
        --datalist ./datalist/train.txt \
        --num_eval 30 \
        --num_points 50000 \
        --noise_min 0.005 --noise_max 0.02 \
        --out_gt ./eval_gt \
        --out_noisy ./eval_noisy \
        --out_mesh ./eval_mesh_normalized \
        --seed 2024

Output layout (matches what evaluate.py expects):
    eval_gt/<synset>/<model_id>/clean.npy
    eval_noisy/<synset>/<model_id>/noisy.npy
    eval_mesh_normalized/<synset>/<model_id>/models/model_normalized.obj
"""

import os
import argparse
import shutil
import numpy as np
import trimesh


def normalize_unit_sphere(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc ** 2).sum(axis=1).max())
    return (pc / scale).astype(np.float32), center.astype(np.float32), np.float32(scale)


def sample_mesh_surface(path, num_samples):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    pts, _ = trimesh.sample.sample_surface(mesh, num_samples)
    return np.asarray(pts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                         help='dataset_train root (contains shapenet/<synset>/<model_id>/models/model_normalized.obj)')
    parser.add_argument('--datalist', type=str, required=True,
                         help='datalist/train.txt (one "shapenet/<synset>/<model_id>" per line)')
    parser.add_argument('--mesh_name', type=str, default='models/model_normalized.obj')
    parser.add_argument('--num_eval', type=int, default=30)
    parser.add_argument('--num_points', type=int, default=50000)
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--out_gt', type=str, default='./eval_gt')
    parser.add_argument('--out_noisy', type=str, default='./eval_noisy')
    parser.add_argument('--out_mesh', type=str, default='./eval_mesh_normalized')
    parser.add_argument('--meta_csv', type=str, default='./eval_meta.csv',
                         help='where to save synset/model_id/noise_std per sample')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    with open(args.datalist, 'r') as f:
        entries = [ln.strip() for ln in f if ln.strip()]

    if args.num_eval > len(entries):
        raise ValueError(f'--num_eval ({args.num_eval}) exceeds datalist size ({len(entries)})')

    chosen = rng.choice(entries, size=args.num_eval, replace=False)

    print(f'Selected {len(chosen)} models out of {len(entries)} in datalist.')

    meta_rows = []

    for rel in chosen:
        # rel looks like: shapenet/<synset>/<model_id>
        parts = rel.split('/')
        synset, model_id = parts[-2], parts[-1]

        mesh_path = os.path.join(args.root, rel, args.mesh_name)
        if not os.path.exists(mesh_path):
            print(f'  [skip] mesh not found: {mesh_path}')
            continue

        # 1. Sample clean points from mesh, normalize to unit sphere
        #    (keep center/scale -- we need to apply the SAME transform to the
        #    mesh, since evaluate.py re-normalizes the mesh using clean.npy's
        #    OWN bbox, which is a no-op once clean.npy is already normalized.
        #    If we hand it a raw-scale mesh, the P2S distances come out wrong.)
        pc = sample_mesh_surface(mesh_path, args.num_points)
        pc_clean, center, scale = normalize_unit_sphere(pc)

        # 2. Add Laplace noise, same distribution as training/test generation
        noise_std = rng.uniform(args.noise_min, args.noise_max)
        noise = rng.laplace(0, noise_std, size=pc_clean.shape).astype(np.float32)
        pc_noisy = (pc_clean + noise).astype(np.float32)

        # 3. Save clean.npy
        gt_dir = os.path.join(args.out_gt, synset, model_id)
        os.makedirs(gt_dir, exist_ok=True)
        np.save(os.path.join(gt_dir, 'clean.npy'), pc_clean)

        # 4. Save noisy.npy
        noisy_dir = os.path.join(args.out_noisy, synset, model_id)
        os.makedirs(noisy_dir, exist_ok=True)
        np.save(os.path.join(noisy_dir, 'noisy.npy'), pc_noisy)

        # 5. Normalize the mesh with the SAME center/scale as pc_clean, then save it
        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float32) - center) / scale

        mesh_out_dir = os.path.join(args.out_mesh, synset, model_id, 'models')
        os.makedirs(mesh_out_dir, exist_ok=True)
        mesh.export(os.path.join(mesh_out_dir, 'model_normalized.obj'))

        print(f'  [ok] {rel}  noise_std={noise_std:.4f}')
        meta_rows.append((f'{synset}/{model_id}', synset, model_id, noise_std))

    import csv
    with open(args.meta_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['key', 'category', 'model_id', 'noise_std'])
        writer.writerows(meta_rows)
    print(f'Metadata (noise_std per sample) saved to: {args.meta_csv}')

    print('Done.')


if __name__ == '__main__':
    main()