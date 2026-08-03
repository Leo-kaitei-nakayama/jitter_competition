"""
Verify and benchmark the vectorized farthest-point sampling.

The old FPS pulled the running `farthest` index to the host with .item() on
every sampled point. Downsampling calls FPS once per encoder block per batch
element, so at the default settings (1000-point patches, stride_list=[4,3,2,1],
batch 8) a single forward pass paid roughly 16,000 blocking GPU syncs:

    block 1: 1000 -> 800 samples   x8 batch =  6400 syncs
    block 2:  800 -> 600 samples   x8 batch =  4800 syncs
    block 3:  600 -> 400 samples   x8 batch =  3200 syncs
    block 4:  400 -> 200 samples   x8 batch =  1600 syncs

Sync latency rather than arithmetic dominated the step time, which is also why
adding GPUs did not make training faster -- every rank pays the same 16k syncs.

This script checks two things:
  1. CORRECTNESS -- the vectorized version returns exactly the same indices as
     the original loop. If this fails, do not use the new code.
  2. SPEED -- how much wall-clock the change actually buys on your hardware.

Usage:
    python bench_fps.py                      # defaults matching training
    python bench_fps.py --batch 8 --n 1000 --cpu
"""
import argparse
import time

import numpy as np
import jittor as jt

from models.pointops_jt import (
    _fps_batched, _fps_reference, furthestsampling, farthest_point_sampling,
)


def timed(fn, warmup=1, repeat=3):
    """Run fn, forcing a sync so lazy execution cannot hide the cost."""
    for _ in range(warmup):
        out = fn()
        jt.sync_all(True)
    best = float('inf')
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        jt.sync_all(True)
        best = min(best, time.perf_counter() - t0)
    return out, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--n', type=int, default=1000, help='points per patch')
    parser.add_argument('--stride_list', type=int, nargs='+', default=[4, 3, 2, 1])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    jt.flags.use_cuda = 0 if args.cpu else 1
    print(f'device: {"cpu" if args.cpu else "cuda"}   batch={args.batch}  n={args.n}')

    rng = np.random.RandomState(args.seed)
    pts_np = rng.randn(args.batch, args.n, 3).astype(np.float32)
    pts = jt.array(pts_np)

    # ---------------- correctness ----------------
    print('\n' + '=' * 62)
    print('CORRECTNESS: vectorized vs original loop')
    print('=' * 62)
    n_sample = args.n * args.stride_list[0] // (args.stride_list[0] + 1)

    new_idx = _fps_batched(pts, n_sample).numpy()
    ref_idx = np.stack([_fps_reference(pts[b], n_sample).numpy()
                        for b in range(args.batch)], axis=0)

    exact = np.array_equal(new_idx, ref_idx)
    print(f'  indices identical: {exact}')
    if not exact:
        n_diff = int((new_idx != ref_idx).sum())
        print(f'  !! {n_diff}/{new_idx.size} indices differ -- DO NOT USE, report this')
        first = np.argwhere(new_idx != ref_idx)[0]
        b, i = int(first[0]), int(first[1])
        print(f'  first mismatch at batch {b} step {i}: '
              f'new={new_idx[b, i]} ref={ref_idx[b, i]}')
    else:
        print('  -> safe to use; no retraining needed, the sampling is unchanged')

    # offset-format wrapper must agree too
    o = jt.array(np.array([(i + 1) * args.n for i in range(args.batch)], dtype=np.int32))
    n_o = jt.array(np.array([(i + 1) * n_sample for i in range(args.batch)], dtype=np.int32))
    flat = furthestsampling(pts.reshape(-1, 3), o, n_o).numpy()
    expect = (ref_idx + (np.arange(args.batch) * args.n)[:, None]).reshape(-1)
    print(f'  furthestsampling offset wrapper matches: {np.array_equal(flat, expect)}')

    sampled, idx_list = farthest_point_sampling(pts, 64)
    print(f'  farthest_point_sampling shapes: sampled={tuple(sampled.shape)} '
          f'len(indices)={len(idx_list)} indices[0]={tuple(idx_list[0].shape)}')

    # ---------------- speed ----------------
    print('\n' + '=' * 62)
    print('SPEED: one full encoder pass worth of FPS calls')
    print('=' * 62)
    print(f'{"block":>6} {"in":>6} {"out":>6} {"old (s)":>10} {"new (s)":>10} {"speedup":>9}')

    total_old = total_new = 0.0
    cur = args.n
    for bi, stride in enumerate(args.stride_list):
        out_n = cur * stride // (stride + 1)
        blk = jt.array(rng.randn(args.batch, cur, 3).astype(np.float32))

        _, t_new = timed(lambda: _fps_batched(blk, out_n))
        _, t_old = timed(lambda: [_fps_reference(blk[b], out_n) for b in range(args.batch)],
                         warmup=0, repeat=1)

        total_old += t_old
        total_new += t_new
        print(f'{bi:>6} {cur:>6} {out_n:>6} {t_old:>10.3f} {t_new:>10.3f} '
              f'{t_old / max(t_new, 1e-9):>8.1f}x')
        cur = out_n

    print('-' * 62)
    print(f'{"total":>6} {"":>6} {"":>6} {total_old:>10.3f} {total_new:>10.3f} '
          f'{total_old / max(total_new, 1e-9):>8.1f}x')
    print()
    print(f'FPS cost per training step: {total_old:.2f}s -> {total_new:.2f}s')
    print('(inference gains too -- patch_based_denoise runs farthest_point_sampling')
    print(' over every patch of every cloud.)')


if __name__ == '__main__':
    main()
