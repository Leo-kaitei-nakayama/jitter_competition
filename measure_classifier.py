"""
Is the trained classifier (ScaleNet) good enough to drive adaptive depth?

Run this BEFORE retraining the backbone with adaptive depth. It answers, from
data rather than theory, the three questions the retrain decision hangs on:

  1. RANGE. ScaleNet ends in tanh, so its output rho lives in (-1, 1). The
     training target is the noisy/clean voxel-entropy ratio, which nothing
     forces below 1. If most targets sit outside (-1, 1), the network has been
     chasing values it cannot emit and its output is saturated, not informative.

  2. SIGNAL. Adaptive depth only helps if rho tracks the actual noise level.
     Reported as Pearson correlation of (prediction, target) and of
     (prediction, true sigma), plus a per-sigma-bin table you can eyeball for
     monotonicity.

  3. EFFECT. Even a perfect rho only matters through the depth formula
     (models/feature.py:assign_n_layer_based_on_rho). The depth histogram shows
     what the classifier would actually DO: if every patch lands on the same
     depth, adaptive depth is a no-op regardless of prediction quality.

Reads only training meshes (same pipeline as train_classifier_on_starter.py)
plus the classifier checkpoint. Touches nothing, writes nothing.

Usage:
    python measure_classifier.py \
        --classify_ckpt experiments/classify/classify-epoch049.pkl \
        --data_root ./dataset_train --datalist ./datalist/train.txt \
        --n_batches 40
"""
import math
import argparse

import numpy as np
import jittor as jt

from models.classify import Classify
from models.utils import get_entropy_B
from bridge.data_bridge import ShapeNetPatchTrainDataset

jt.flags.use_cuda = 1


def depth_from_rho(rho, L=4, gamma=0.396):
    """Mirror of models/feature.py:assign_n_layer_based_on_rho."""
    return min(L, max(2, math.ceil(L - (L - 1) * math.log(gamma * float(rho) + 1))))


def main(args):
    model = Classify(frame_knn=args.frame_knn)
    model.load(args.classify_ckpt)
    model.eval()
    print(f'classifier: {args.classify_ckpt}')

    loader = ShapeNetPatchTrainDataset(
        root=args.data_root, datalist=args.datalist,
        num_samples=args.num_samples, patch_size=args.patch_size,
        noise_min=args.noise_min, noise_max=args.noise_max,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, noise_dist=args.noise_dist,
        mesh_name=args.mesh_name,
    )

    preds, targets, sigmas = [], [], []
    with jt.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.n_batches:
                break
            noisy, clean = batch['pcl_noisy'], batch['pcl_clean']
            seeds, std = batch['seed_pnts'], batch['pcl_std']
            N, M = noisy.shape[1], clean.shape[1]

            # identical centering to Classify.get_supervised_loss_nn
            noisy = noisy - seeds.repeat(1, N, 1)
            clean = clean - seeds.repeat(1, M, 1)

            target = (get_entropy_B(noisy) / get_entropy_B(clean)).numpy().reshape(-1)
            pre, _ = model.feature_nets[0](noisy, None)
            pred = pre.numpy().reshape(-1)

            preds.append(pred)
            targets.append(target)
            sigmas.append(std.numpy().reshape(-1))

    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    sigma = np.concatenate(sigmas)
    n = len(pred)
    print(f'\n{n} patches, sigma in [{sigma.min():.4f}, {sigma.max():.4f}]')

    # ---- 1. RANGE ----
    out_frac = float((np.abs(target) >= 1.0).mean())
    print('\n--- 1. range (the tanh question) ---')
    print(f'target : min={target.min():.3f} mean={target.mean():.3f} '
          f'max={target.max():.3f}')
    print(f'pred   : min={pred.min():.3f} mean={pred.mean():.3f} '
          f'max={pred.max():.3f}  (tanh caps at +/-1)')
    print(f'targets outside tanh range: {100*out_frac:.1f}%')
    if out_frac > 0.5:
        print('  => MOST targets are unreachable. The classifier is saturated;')
        print('     fix the output activation (or rescale the target) and retrain')
        print('     it before trusting adaptive depth.')

    # ---- 2. SIGNAL ----
    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else float('nan')

    print('\n--- 2. signal ---')
    print(f'corr(pred, target) = {pearson(pred, target):+.3f}')
    print(f'corr(pred, sigma)  = {pearson(pred, sigma):+.3f}   '
          f'(target-vs-sigma = {pearson(target, sigma):+.3f}, the ceiling)')

    edges = np.linspace(args.noise_min, args.noise_max, 5)
    print('\n  sigma bin        n    target(mean)  pred(mean)')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sigma >= lo) & (sigma <= hi)
        if m.sum() == 0:
            continue
        print(f'  {lo:.4f}-{hi:.4f}  {m.sum():4d}    {target[m].mean():8.3f}    '
              f'{pred[m].mean():8.3f}')

    # ---- 3. EFFECT ----
    print('\n--- 3. what depths would be chosen ---')
    for name, arr in (('pred', pred), ('target (perfect classifier)', target)):
        depths = np.array([depth_from_rho(max(r, -0.99)) for r in arr])
        counts = {d: int((depths == d).sum()) for d in (2, 3, 4)}
        print(f'  from {name:28s}: ' +
              '  '.join(f'depth{d}: {100*c/n:5.1f}%' for d, c in counts.items()))
    print('\nIf everything lands on one depth, adaptive depth is a no-op:')
    print('retraining the backbone for it would buy nothing.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--classify_ckpt', type=str, required=True)
    p.add_argument('--data_root', type=str, default='./dataset_train')
    p.add_argument('--datalist', type=str, default='./datalist/train.txt')
    p.add_argument('--n_batches', type=int, default=40,
                   help='batches to measure; 40 x 8 = 320 patches is plenty')
    p.add_argument('--num_samples', type=int, default=32768)
    p.add_argument('--patch_size', type=int, default=1000)
    p.add_argument('--noise_min', type=float, default=0.005)
    p.add_argument('--noise_max', type=float, default=0.02)
    p.add_argument('--noise_dist', type=str, default='laplace',
                   choices=['laplace', 'gaussian'])
    p.add_argument('--mesh_name', type=str, default='models/model_normalized.obj')
    p.add_argument('--frame_knn', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=2024)
    args = p.parse_args()

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
