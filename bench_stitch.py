"""
Verify the linear-memory stitching path end-to-end, on real KNN geometry.

Two claims are checked, both against the exact code they replaced:

  1. knn_points chunking: forcing tiny chunks must give bit-identical
     indices/distances to the single-shot matrix.
  2. select_stitch_source: on the pid/pdist a real cloud produces, the sparse
     selection must match the dense argmax construction exactly.
     (The pure-numpy logic is also fuzz-tested against ties separately; this
     re-checks it on real float distributions.)

Plus a memory table: what the dense arrays would have cost at this N and at
round-B-plausible sizes.

Needs jittor but no checkpoint and no dataset -- the cloud is synthetic.

Usage:
    python bench_stitch.py                 # N=50000, the round-A size
    python bench_stitch.py --N 200000      # round-B-plausible
"""
import argparse
import time

import numpy as np
import jittor as jt

from models.denoiseCD import DenoiseNetCD
from models.pointops_jt import knn_points, farthest_point_sampling

jt.flags.use_cuda = 1


def dense_reference(pid_np, pdist_np, N):
    """The stitching selection exactly as denoiseCD.py used to build it."""
    num_patches, patch_size = pid_np.shape
    all_dists_np = np.full((num_patches, N), np.inf, dtype=np.float32)
    for pi in range(num_patches):
        all_dists_np[pi, pid_np[pi]] = pdist_np[pi]
    weights = np.exp(-1 * all_dists_np)
    best_weights_idx = weights.argmax(axis=0)
    pos_map = np.full((num_patches, N), -1, dtype=np.int64)
    col = np.arange(patch_size, dtype=np.int64)
    for pi in range(num_patches):
        pos_map[pi, pid_np[pi]] = col
    point_ids = np.arange(N, dtype=np.int64)
    gather_pos = pos_map[best_weights_idx, point_ids]
    covered = gather_pos >= 0
    return (best_weights_idx[covered].astype(np.int32),
            gather_pos[covered].astype(np.int32))


def main(args):
    rng = np.random.default_rng(args.seed)
    # noisy sphere-ish cloud, unit-sphere scale like real inputs
    pc = rng.normal(size=(args.N, 3)).astype(np.float32)
    pc /= np.linalg.norm(pc, axis=1, keepdims=True)
    pc += rng.normal(scale=0.01, size=pc.shape).astype(np.float32)
    pcl = jt.array(pc).unsqueeze(0)

    num_patches = int(args.seed_k * args.N / args.patch_size)
    print(f'N={args.N}, {num_patches} patches of {args.patch_size}')

    t0 = time.time()
    seeds, _ = farthest_point_sampling(pcl, num_patches)
    print(f'FPS: {time.time()-t0:.1f}s')

    # ---- 1. chunked knn == single-shot knn ----
    t0 = time.time()
    d1, i1, _ = knn_points(seeds, pcl, K=args.patch_size)
    d1n, i1n = d1.numpy(), i1.numpy()
    t1 = time.time() - t0
    t0 = time.time()
    d2, i2, _ = knn_points(seeds, pcl, K=args.patch_size,
                           max_chunk_elems=args.N * 8)  # forces many chunks
    d2n, i2n = d2.numpy(), i2.numpy()
    t2 = time.time() - t0
    assert np.array_equal(i1n, i2n), 'chunked knn returned different indices!'
    assert np.array_equal(d1n, d2n), 'chunked knn returned different distances!'
    print(f'1. knn chunking: identical  (single-shot {t1:.1f}s, chunked {t2:.1f}s)')

    # ---- 2. sparse selection == dense reference ----
    pid_np = i1n[0]
    pdist_np = d1n[0]
    pdist_np = pdist_np / pdist_np[:, -1:]

    t0 = time.time()
    sp, sc = DenoiseNetCD.select_stitch_source(pid_np, pdist_np)
    ts = time.time() - t0

    dense_bytes = num_patches * args.N * (4 + 4 + 8)  # all_dists + weights + pos_map
    if dense_bytes > args.dense_limit_gb * 2 ** 30:
        print(f'2. dense reference skipped: it would need '
              f'{dense_bytes/2**30:.1f} GB (that is the point). '
              f'Sparse selection ran in {ts:.2f}s.')
    else:
        t0 = time.time()
        rp, rc = dense_reference(pid_np, pdist_np, args.N)
        td = time.time() - t0
        assert np.array_equal(sp, rp), 'selection chose different patches!'
        assert np.array_equal(sc, rc), 'selection chose different positions!'
        print(f'2. stitch selection: identical, {len(sp)}/{args.N} covered  '
              f'(dense {td:.2f}s, sparse {ts:.2f}s)')

    # ---- 3. memory table ----
    print('\n   stitching bookkeeping, dense (old) vs sparse (new):')
    for n in (50_000, 100_000, 200_000, 500_000):
        p = int(args.seed_k * n / args.patch_size)
        old = p * n * 16 / 2 ** 30
        new = args.seed_k * n * 12 / 2 ** 30  # pid+pdist copies + selection
        print(f'   N={n:>7,}: {old:8.2f} GB  ->  {new:.3f} GB')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=50000)
    p.add_argument('--patch_size', type=int, default=1000)
    p.add_argument('--seed_k', type=int, default=6)
    p.add_argument('--dense_limit_gb', type=float, default=8.0,
                   help='skip the dense reference above this projected size')
    p.add_argument('--seed', type=int, default=2024)
    args = p.parse_args()
    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    main(args)
