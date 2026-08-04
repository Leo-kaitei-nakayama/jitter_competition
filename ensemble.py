"""
Weighted average of several prediction sets.

Different checkpoints make different mistakes. Averaging their outputs cancels
part of the error that is specific to each, which is why the ensemble beat both
of its parents on the tune set (77.96 against 77.26 and 76.26) even though one
parent scored a point WORSE than the other on its own.

This works here only because every prediction set comes from the same input
clouds in the same order, so row i of one file corresponds to row i of another.
That is guaranteed as long as each set was produced by predict_on_starter.py
from the same --data_root and --datalist. The script verifies it by shape and
refuses to mix files that disagree.

Point count is preserved exactly, and output is float32 for the submission
format. Run postprocess.py afterwards if you want the repulsion filter.

Usage:
    python ensemble.py --out_root ./eval_predict/ens \
        --pred_roots ./eval_predict/L3 ./eval_predict/pf_ss1.2 \
        --weights 0.7 0.3
"""
import argparse
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_roots', nargs='+', required=True,
                        help='two or more prediction directories to blend')
    parser.add_argument('--weights', type=float, nargs='+',
                        help='one per --pred_roots; normalised automatically. '
                             'Omit for a plain average.')
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--pred_filename', default='denoised.npy')
    args = parser.parse_args()

    roots = args.pred_roots
    if len(roots) < 2:
        raise SystemExit('need at least two --pred_roots')

    w = np.array(args.weights if args.weights else [1.0] * len(roots), dtype=np.float64)
    if len(w) != len(roots):
        raise SystemExit(f'{len(w)} weights for {len(roots)} roots')
    w = w / w.sum()

    print('blending:')
    for r, wi in zip(roots, w):
        print(f'  {wi:.3f}  {r}')

    base = roots[0]
    files = sorted(glob.glob(os.path.join(base, '**', args.pred_filename),
                             recursive=True))
    if not files:
        raise SystemExit(f'no {args.pred_filename} under {base}')

    written = skipped = 0
    for pa in files:
        rel = os.path.relpath(pa, base)
        parts = [np.load(pa).astype(np.float64)]
        ok = True
        for r in roots[1:]:
            p = os.path.join(r, rel)
            if not os.path.exists(p):
                ok = False
                break
            arr = np.load(p).astype(np.float64)
            if arr.shape != parts[0].shape:
                ok = False
                break
            parts.append(arr)
        if not ok:
            skipped += 1
            continue

        out = sum(wi * a for wi, a in zip(w, parts))
        assert out.shape == parts[0].shape

        dst = os.path.join(args.out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.save(dst, out.astype(np.float32))
        written += 1

    print(f'wrote {written} clouds to {args.out_root}')
    if skipped:
        print(f'WARNING: skipped {skipped} clouds -- missing from some root or '
              f'shape mismatch. Those will score 0 in evaluate.py.')


if __name__ == '__main__':
    main()
