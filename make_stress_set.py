"""
Build a synthetic stress set that probes specific failure modes.

The real eval set only says how the model scores on average, on ShapeNet
furniture, at the noise levels it trained on. It cannot say *why* a sample
scores badly, because every shape varies in a dozen ways at once.

Here each cloud isolates one property. The geometry is generated, so the
ground truth is exact and the mesh is available for P2S, and every shape is
tested under the same noise conditions -- including conditions the model never
trained on, which is what the B-board's "more categories, higher noise" means
in practice.

Shapes (what each one asks)
    box          can it hold a sharp 90-degree edge, or does it round it?
    thin_plate   does a thin structure collapse? The paper calls this the
                 characteristic failure of score-based denoisers.
    sphere       baseline: smooth, uniform curvature, no features to lose
    cylinder     flat, curved and a sharp rim in one shape
    cone         a single sharp apex -- the hardest kind of feature
    torus        curvature that changes sign, convex to concave

Conditions (what each one asks)
    lo / mid / hi    inside the training range (sigma 0.005 / 0.0125 / 0.02)
    extreme          sigma 0.03, 1.5x above anything it trained on
    gaussian         same sigma, different distribution shape
    outlier          mid noise plus 1% of points thrown far off the surface

Output matches the eval layout, so predict_on_starter.py, evaluate.py and
analysis.py all work on it unchanged. analysis.py groups by "category", which
here is the shape name.

    stress_gt/<shape>/<condition>/clean.npy
    stress_noisy/<shape>/<condition>/noisy.npy
    stress_mesh/<shape>/<condition>/models/model_normalized.obj

Usage:
    python make_stress_set.py --num_points 50000
    find stress_noisy -mindepth 2 -maxdepth 2 -type d | sed 's|^stress_noisy/||' \
        | sort > datalist/stress.txt
"""
import argparse
import math
import os

import numpy as np
import trimesh


def build_shapes():
    """Unit-ish primitives, each isolating one geometric property."""
    shapes = {}
    shapes['box'] = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    shapes['thin_plate'] = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
    shapes['sphere'] = trimesh.creation.icosphere(subdivisions=4, radius=0.6)
    shapes['cylinder'] = trimesh.creation.cylinder(radius=0.4, height=1.0, sections=64)
    shapes['cone'] = trimesh.creation.cone(radius=0.5, height=1.0, sections=64)
    try:
        shapes['torus'] = trimesh.creation.torus(major_radius=0.45, minor_radius=0.15)
    except AttributeError:
        pass          # older trimesh has no torus; the rest still probe plenty
    return shapes


def normalize_unit_sphere(pc):
    p_max, p_min = pc.max(axis=0), pc.min(axis=0)
    center = (p_max + p_min) / 2
    pc = pc - center
    scale = np.sqrt((pc ** 2).sum(axis=1).max())
    return (pc / scale).astype(np.float32), center.astype(np.float32), np.float32(scale)


def add_noise(clean, kind, sigma, rng):
    """Every kind is parameterised by its STANDARD DEVIATION, not a scale."""
    n = clean.shape
    if kind == 'gaussian':
        return rng.normal(0.0, sigma, size=n).astype(np.float32)
    if kind == 'laplace':
        return rng.laplace(0.0, sigma / math.sqrt(2.0), size=n).astype(np.float32)
    if kind == 'uniform':
        h = sigma * math.sqrt(3.0)          # uniform(-h,h) has std h/sqrt(3)
        return rng.uniform(-h, h, size=n).astype(np.float32)
    if kind == 'outlier':
        base = rng.laplace(0.0, sigma / math.sqrt(2.0), size=n).astype(np.float32)
        hit = rng.random(n[0]) < 0.01       # 1% thrown far off the surface
        base[hit] += rng.normal(0.0, sigma * 12.0, size=(hit.sum(), 3)).astype(np.float32)
        return base
    raise ValueError(kind)


CONDITIONS = [
    # name        kind        sigma    trained on?
    ('lo',       'laplace',   0.005),   # yes, bottom of range
    ('mid',      'laplace',   0.0125),  # yes, middle
    ('hi',       'laplace',   0.020),   # yes, top of range
    ('extreme',  'laplace',   0.030),   # NO -- 1.5x beyond training
    ('gaussian', 'gaussian',  0.0125),  # NO -- different distribution
    ('outlier',  'outlier',   0.0125),  # NO -- gross outliers
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_points', type=int, default=50000)
    parser.add_argument('--out_gt', default='./stress_gt')
    parser.add_argument('--out_noisy', default='./stress_noisy')
    parser.add_argument('--out_mesh', default='./stress_mesh')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    shapes = build_shapes()
    rows = []

    for name, mesh in shapes.items():
        pts, _ = trimesh.sample.sample_surface(mesh, args.num_points,
                                               seed=rng.randint(1 << 30))
        clean, center, scale = normalize_unit_sphere(np.asarray(pts, dtype=np.float32))

        m = mesh.copy()
        m.vertices = (np.asarray(m.vertices, dtype=np.float32) - center) / scale

        for cond, kind, sigma in CONDITIONS:
            noisy = (clean + add_noise(clean, kind, sigma, rng)).astype(np.float32)

            for root, sub, data in [
                (args.out_gt, 'clean.npy', clean),
                (args.out_noisy, 'noisy.npy', noisy),
            ]:
                d = os.path.join(root, name, cond)
                os.makedirs(d, exist_ok=True)
                np.save(os.path.join(d, sub), data)

            md = os.path.join(args.out_mesh, name, cond, 'models')
            os.makedirs(md, exist_ok=True)
            m.export(os.path.join(md, 'model_normalized.obj'))

            rows.append((f'{name}/{cond}', name, cond, kind, sigma))
            print(f'  {name:<11} {cond:<9} {kind:<9} sigma={sigma:.4f}')

    import csv
    with open('stress_meta.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'category', 'model_id', 'noise_kind', 'noise_std'])
        w.writerows(rows)

    print(f'\n{len(rows)} clouds across {len(shapes)} shapes x {len(CONDITIONS)} conditions')
    print('metadata: stress_meta.csv')
    print('\nnext:')
    print("  find stress_noisy -mindepth 2 -maxdepth 2 -type d "
          "| sed 's|^stress_noisy/||' | sort > datalist/stress.txt")


if __name__ == '__main__':
    main()
