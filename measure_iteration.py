"""
Per-point iteration dynamics: is per-point early stopping worth building?

Runs the real model on real (tune-split) eval clouds, records every point's
position after every diffusion step, and answers three questions with numbers:

  1. OVERSHOOT. What fraction of points get WORSE from step to step? If points
     move together, a global L is fine and per-point stopping buys nothing.

  2. ORACLE GAIN. If every point could stop at its own best step, how much
     lower would the mean error be than stopping everything at L? This is the
     hard upper bound on any per-point stopping rule -- the analog of
     check_learnable.py's 12% bound for the refine head.

  3. PREDICTABILITY. Does the per-point score norm at step t predict whether
     step t+1 will hurt? A stopping rule needs a signal available at inference;
     this is the natural candidate (same family as the σ̂ estimate).

Also decomposes each step's movement into a productive component (toward the
nearest clean point) and a drift component (orthogonal to it) -- the measured
version of StraightPCF's "curved trajectories accumulate error" claim, on our
own model.

Uses the tune split only. Touches nothing, writes nothing.

Usage:
    python measure_iteration.py \
        --ckpt experiments/asdn_diffusion/asdn-epoch049.pkl \
        --use_fusion --static_depth \
        --noisy_root ./eval_noisy_tune --clean_root ./eval_gt_tune \
        --n_clouds 6 --chunks 2
"""
import os
import argparse

import numpy as np
import jittor as jt
from scipy.spatial import cKDTree

from models.denoiseCD import DenoiseNetCD
from models.pointops_jt import knn_points, farthest_point_sampling
from bridge.data_bridge import normalize_unit_sphere

jt.flags.use_cuda = 1


def find_keys(root, name='noisy.npy'):
    keys = []
    for dirpath, _, files in os.walk(root):
        if name in files:
            keys.append(os.path.relpath(dirpath, root))
    return sorted(keys)


def main(args):
    model = DenoiseNetCD(fusion_k=args.fusion_k, fusion_gate=args.fusion_gate,
                         fusion_include_self=args.fusion_include_self,
                         static_depth=args.static_depth,
                         max_sigma=args.max_sigma)
    if args.use_fusion:
        model.feature_nets.use_fusion = True
    model.load(args.ckpt)
    model.eval()

    keys = find_keys(args.noisy_root)[:args.n_clouds]
    if not keys:
        raise SystemExit(f'no noisy.npy under {args.noisy_root}')
    print(f'{len(keys)} clouds from {args.noisy_root}')

    L = args.diffusion_L
    # per step-index accumulators, step 0 = state before any diffusion step
    errs = [[] for _ in range(L + 1)]
    worse = [[] for _ in range(L)]          # err[t+1] > err[t] per point
    prod = [[] for _ in range(L)]           # movement toward nearest clean point
    drift = [[] for _ in range(L)]          # movement orthogonal to that
    norm_vs_gain = [[] for _ in range(L)]   # (score_norm_t, err_t+1 - err_t)
    best_step = []

    for key in keys:
        noisy = np.load(os.path.join(args.noisy_root, key, 'noisy.npy')).astype(np.float32)
        clean = np.load(os.path.join(args.clean_root, key, 'clean.npy')).astype(np.float32)
        pc_norm, center, scale = normalize_unit_sphere(noisy)
        clean_n = (clean - center) / scale
        tree = cKDTree(clean_n)

        pcl = jt.array(pc_norm).unsqueeze(0)
        N = pc_norm.shape[0]
        num_patches = int(args.seed_k * N / args.patch_size)
        seeds, _ = farthest_point_sampling(pcl, num_patches)
        _, _, patches = knn_points(seeds, pcl, K=args.patch_size, return_nn=True)
        patches = patches[0]
        seeds_rep = seeds.squeeze(0).unsqueeze(1).repeat(1, args.patch_size, 1)
        patches = patches - seeds_rep

        patch_step = int(N / (args.seed_k_alpha * args.patch_size))
        chunk_starts = list(range(0, num_patches, patch_step))

        # --- tau exactly as inference estimates it (probe pass + Eq. 15/16) ---
        probe_frac = float(args.diffusion_t_start) / model.schedule.T
        norms = []
        with jt.no_grad():
            for s in chunk_starts:
                sc, _ = model._patch_forward(patches[s:s + patch_step],
                                             feat_T=None, t_frac_val=probe_frac)
                norms.append(jt.sqrt((sc ** 2).sum(dim=-1) + 1e-12).numpy().reshape(-1))
        tau = model.schedule.estimate_tau(np.concatenate(norms),
                                          sigma_scale=args.sigma_scale,
                                          estimator=args.sigma_estimator)
        tau = int(min(max(tau, L), model.schedule.T))

        # --- instrumented diffusion on the first `chunks` chunks ---
        m = args.mask_size
        for s in chunk_starts[:args.chunks]:
            chunk = patches[s:s + patch_step]
            seed_off = seeds_rep[s:s + patch_step]
            _, traj = model.denoise_langevin_dynamics_diffusion(
                chunk, L=L, t_start=tau, t_norm=args.t_norm, return_traj=True)

            pos = [ (chunk + seed_off).numpy()[:, :m, :] ]
            snorm = []
            for p, nv in traj:
                pos.append(p[:, :m, :] + seed_off.numpy()[:, :m, :])
                snorm.append(nv[:, :m])

            e = []
            for p in pos:
                d, i = tree.query(p.reshape(-1, 3))
                e.append(d)
            nn0 = [tree.query(p.reshape(-1, 3))[1] for p in pos[:-1]]

            for t in range(L + 1):
                errs[t].append(e[t])
            for t in range(L):
                worse[t].append(e[t + 1] > e[t])
                # decompose the step's movement
                cur = pos[t].reshape(-1, 3)
                nxt = pos[t + 1].reshape(-1, 3)
                target = clean_n[nn0[t]]
                u = target - cur
                un = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
                d_vec = nxt - cur
                along = (d_vec * un).sum(axis=1)
                ortho = np.linalg.norm(d_vec - along[:, None] * un, axis=1)
                prod[t].append(along)
                drift[t].append(ortho)
                norm_vs_gain[t].append(
                    np.stack([snorm[t].reshape(-1), e[t + 1] - e[t]]))

            E = np.stack(e, axis=0)           # (L+1, pts)
            best_step.append(E.argmin(axis=0))

        print(f'  {key}: tau={tau}')

    # ---- report ----
    def cat(lst):
        return np.concatenate(lst)

    print(f'\n=== per-step table ({len(cat(errs[0]))} points, mask={args.mask_size}, '
          f'L={L}) ===')
    print('step | mean err   | % worse than prev | move toward | move drift')
    print(f'  0  | {cat(errs[0]).mean():.6f}  |        --         |     --      |    --')
    for t in range(L):
        print(f'  {t+1}  | {cat(errs[t+1]).mean():.6f}  |      {100*cat(worse[t]).mean():5.1f}%       '
              f'| {cat(prod[t]).mean():+.6f}   | {cat(drift[t]).mean():.6f}')

    bs = cat(best_step)
    final = cat(errs[L])
    oracle = np.minimum.reduce([cat(errs[t]) for t in range(L + 1)])
    gain = 100 * (1 - oracle.mean() / final.mean())
    print(f'\nbest-step histogram: ' +
          '  '.join(f'step{t}: {100*(bs==t).mean():.1f}%' for t in range(L + 1)))
    print(f'ORACLE per-point stop: mean err {final.mean():.6f} -> {oracle.mean():.6f} '
          f'({gain:.1f}% lower). This is the ceiling for any stopping rule.')

    print('\npredictability of "next step hurts" from the score norm:')
    for t in range(L):
        a = np.concatenate([x for x in norm_vs_gain[t]], axis=1)
        n, g = a[0], a[1]
        n, g = n - n.mean(), g - g.mean()
        c = (n * g).sum() / np.sqrt((n * n).sum() * (g * g).sum() + 1e-30)
        print(f'  step {t}->{t+1}: corr(score_norm, err change) = {c:+.3f}')

    print('\nHow to read: if the oracle gain is small (<5%), per-point stopping '
          'is closed. If it is large but corr is ~0, the gain is real but '
          'unreachable with this signal. Large gain + strong corr = build it.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--noisy_root', type=str, default='./eval_noisy_tune')
    p.add_argument('--clean_root', type=str, default='./eval_gt_tune')
    p.add_argument('--n_clouds', type=int, default=6)
    p.add_argument('--chunks', type=int, default=2,
                   help='patch chunks per cloud to instrument')
    p.add_argument('--mask_size', type=int, default=256,
                   help='innermost patch points measured (KNN columns are '
                        'seed-distance ordered, so low columns = patch center)')
    p.add_argument('--diffusion_L', type=int, default=3)
    p.add_argument('--diffusion_t_start', type=int, default=632)
    p.add_argument('--sigma_scale', type=float, default=1.5)
    p.add_argument('--sigma_estimator', type=str, default='rms', choices=['var', 'rms'])
    p.add_argument('--t_norm', type=str, default='T', choices=['T', 'tau'])
    p.add_argument('--patch_size', type=int, default=1000)
    p.add_argument('--seed_k', type=int, default=6)
    p.add_argument('--seed_k_alpha', type=int, default=10)
    p.add_argument('--use_fusion', action='store_true')
    p.add_argument('--static_depth', action='store_true')
    p.add_argument('--fusion_k', type=int, default=16)
    p.add_argument('--fusion_gate', type=str, default='pos', choices=['pos', 'posfeat'])
    p.add_argument('--fusion_include_self', action='store_true')
    p.add_argument('--max_sigma', type=float, default=None)
    p.add_argument('--seed', type=int, default=2024)
    args = p.parse_args()

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
