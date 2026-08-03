"""
Sanity-check predictions before zipping for submission.

Checks, for every entry in --datalist:
  1. denoised.npy exists
  2. point count matches the corresponding noisy.npy (this is what caused
     the earlier 0-score submissions -- shape mismatch)
  3. no NaN / Inf values
  4. coordinate magnitude is in a sane range (catches exploded/collapsed
     predictions, e.g. the niters=3 case that scored CD=0)

Usage:
    python check_submission.py \
        --datalist ./datalist/test.txt \
        --pred_root ./results/dataset_test_noisy_final \
        --noisy_root ./dataset_test_noisy \
        --pred_name denoised.npy \
        --noisy_name noisy.npy
"""
import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datalist', type=str, required=True)
    parser.add_argument('--pred_root', type=str, required=True)
    parser.add_argument('--noisy_root', type=str, required=True)
    parser.add_argument('--pred_name', type=str, default='denoised.npy')
    parser.add_argument('--noisy_name', type=str, default='noisy.npy')
    parser.add_argument('--max_abs_coord', type=float, default=10.0,
                         help='flag predictions with |coord| beyond this (unit-sphere data should be ~O(1))')
    args = parser.parse_args()

    with open(args.datalist, 'r') as f:
        entries = [ln.strip() for ln in f if ln.strip()]

    n_total = len(entries)
    missing = []
    shape_mismatch = []
    nan_inf = []
    exploded = []
    ok = []

    for rel in entries:
        pred_path = os.path.join(args.pred_root, rel, args.pred_name)
        noisy_path = os.path.join(args.noisy_root, rel, args.noisy_name)

        if not os.path.exists(pred_path):
            missing.append(rel)
            continue

        pred = np.load(pred_path)
        noisy = np.load(noisy_path) if os.path.exists(noisy_path) else None

        if noisy is not None and pred.shape[0] != noisy.shape[0]:
            shape_mismatch.append((rel, pred.shape[0], noisy.shape[0]))
            continue

        if not np.all(np.isfinite(pred)):
            nan_inf.append(rel)
            continue

        max_abs = float(np.abs(pred).max())
        if max_abs > args.max_abs_coord:
            exploded.append((rel, max_abs))
            continue

        ok.append(rel)

    print('=' * 65)
    print(f'Checked {n_total} entries from {args.datalist}')
    print('=' * 65)
    print(f'  OK:                {len(ok)}')
    print(f'  Missing:           {len(missing)}')
    print(f'  Shape mismatch:    {len(shape_mismatch)}')
    print(f'  NaN/Inf:           {len(nan_inf)}')
    print(f'  Exploded (>|{args.max_abs_coord}|): {len(exploded)}')
    print('=' * 65)

    if missing:
        print(f'\n[MISSING] {len(missing)} entries have no {args.pred_name}:')
        for rel in missing[:20]:
            print(f'  {rel}')
        if len(missing) > 20:
            print(f'  ... and {len(missing) - 20} more')

    if shape_mismatch:
        print(f'\n[SHAPE MISMATCH] {len(shape_mismatch)} entries (pred_n, noisy_n):')
        for rel, pn, nn in shape_mismatch:
            print(f'  {rel}: pred={pn}  noisy={nn}')

    if nan_inf:
        print(f'\n[NAN/INF] {len(nan_inf)} entries:')
        for rel in nan_inf:
            print(f'  {rel}')

    if exploded:
        print(f'\n[EXPLODED / suspicious magnitude] {len(exploded)} entries:')
        for rel, m in exploded:
            print(f'  {rel}: max|coord|={m:.3f}')

    all_good = not (missing or shape_mismatch or nan_inf or exploded)
    print()
    if all_good:
        print('All entries look good -- safe to zip for submission.')
    else:
        print('Found issues above -- fix these before zipping.')


if __name__ == '__main__':
    main()
    