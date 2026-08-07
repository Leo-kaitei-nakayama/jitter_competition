"""
Train the backbone AND the RefineHead in one command.

Why this exists: the production pipeline is train_on_starter.py -> freeze ->
train_refine.py, a two-command sequence with a checkpoint handoff in the
middle. Round B swaps the dataset, and the rule is that the algorithm stays
basically the same -- so the retrain path must be one command on one datalist,
not a relay race. This script is that command.

What it does per step:

    1. the exact train_on_starter.py backbone step (score loss, Eq. 14 target)
    2. [from --head_start_epoch on] roll the CURRENT backbone through the real
       inference path (L diffusion steps, eval mode, no grad) and train the
       RefineHead on the result, exactly as train_refine.py would

Two facts make this safe rather than a new experiment:

  * The head's gradient CANNOT reach the backbone. Its inputs (denoised
    positions, encoder features) are produced under jt.no_grad(), so the
    backbone's training trajectory is the same as train_on_starter.py's --
    joint training changes nothing about the model that already scores 78.99.
  * The head is zero-initialised (see models/refine.py): until trained it is
    exactly the identity, so a joint checkpoint pair is never worse than the
    bare backbone except through head training itself.

The head starts late (--head_start_epoch) for two reasons: rollouts through a
half-trained backbone are noise the zero-init head would chase, and skipping
the rollout keeps early epochs at full backbone speed (the rollout is L+1
extra forward passes, roughly doubling step cost once it starts).

Outputs per save interval: asdn-epochXXX.pkl AND refine-epochXXX.pkl, the same
two files the old pipeline produced -- predict_on_starter.py consumes them
unchanged via --ckpt / --refine_ckpt.

Usage (round-A recipe, from scratch):
    mpirun -np 4 python -u train_joint.py \
        --data_root ./dataset_train --datalist ./datalist/train.txt \
        --use_fusion --static_depth \
        --lr 1e-3 --lr_min 1e-5 --epochs 60 --batch_size 48 \
        --num_workers 2 --save_interval 5 \
        --save_dir experiments/joint

The rejected loss-term flags (--uniformity_weight, --coverage_weight,
--tangent_weight, --exact_score) are deliberately NOT here; they live on in
train_on_starter.py behind default-off flags. See the README's
distribution-terms section for why all of them lost.
"""
import os
import math
import argparse

import numpy as np
import jittor as jt
from tqdm import tqdm

from models.denoiseCD import DenoiseNetCD
from models.refine import RefineHead
from models.InfoCD import chamfer_dist
from bridge.data_bridge import ShapeNetPatchTrainDataset

jt.flags.use_cuda = 1


def cosine_lr(base, floor, epoch, epochs):
    if floor is None:
        return base
    return floor + 0.5 * (base - floor) * (
        1 + math.cos(math.pi * epoch / max(1, epochs - 1)))


def main(args):
    is_master = (jt.rank == 0)
    if is_master:
        os.makedirs(args.save_dir, exist_ok=True)

    # ---- backbone: identical construction to train_on_starter.py ----
    model = DenoiseNetCD(classify_ckpt=args.classify_ckpt,
                         classify_frame_knn=args.classify_frame_knn,
                         fusion_k=args.fusion_k, fusion_gate=args.fusion_gate,
                         fusion_include_self=args.fusion_include_self,
                         static_depth=args.static_depth,
                         max_sigma=args.max_sigma)
    if args.use_fusion:
        model.feature_nets.use_fusion = True
    if args.init_ckpt is not None:
        model.load(args.init_ckpt)
        if is_master:
            print(f'Loaded init weights from {args.init_ckpt}')
        if args.classify_ckpt is not None:
            model.feature_nets.classify.load(args.classify_ckpt)
            model.feature_nets.classify.eval()
            if is_master:
                print(f'Re-applied classifier weights from {args.classify_ckpt}')
    if args.no_bn_sync:
        n_bn = 0
        for m in model.modules():
            if hasattr(m, 'sync'):
                m.sync = False
                n_bn += 1
        if is_master:
            print(f'Disabled BatchNorm sync on {n_bn} modules')

    # ---- head: identical construction to train_refine.py ----
    head = RefineHead(feat_dim=args.feat_dim, hidden=args.hidden,
                      k=args.head_k, n_layers=args.head_layers)
    if args.init_head is not None:
        head.load(args.init_head)

    # Two optimizers, on purpose. The two losses touch disjoint parameter
    # sets (the head's inputs are detached), so a shared optimizer would only
    # entangle their learning rates. Separate ones keep the backbone step
    # bit-compatible with train_on_starter.py and let the head keep
    # train_refine.py's own lr.
    opt_b = jt.optim.Adam(model.feature_nets.parameters(), lr=args.lr)
    opt_h = jt.optim.Adam(head.parameters(), lr=args.head_lr)

    loader = ShapeNetPatchTrainDataset(
        root=args.data_root, datalist=args.datalist,
        num_samples=args.num_samples, patch_size=args.patch_size,
        noise_min=args.noise_min, noise_max=args.noise_max,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, noise_dist=args.noise_dist,
        mesh_name=args.mesh_name,
    )

    sched = model.schedule
    use_mask = 0 < args.mask_size < args.patch_size
    M = args.mask_size if use_mask else args.patch_size

    for epoch in range(args.epochs):
        opt_b.lr = cosine_lr(args.lr, args.lr_min, epoch, args.epochs)
        head_on = epoch >= args.head_start_epoch
        if head_on:
            # the head's own cosine, over ITS training window, so a late start
            # does not rob it of its high-lr phase
            opt_h.lr = cosine_lr(args.head_lr, args.lr_min,
                                 epoch - args.head_start_epoch,
                                 args.epochs - args.head_start_epoch)
        if is_master:
            msg = f'epoch {epoch}: lr = {opt_b.lr:.2e}'
            if head_on:
                msg += f', head lr = {opt_h.lr:.2e}'
            print(msg)

        b_losses, h_losses, h_cds = [], [], []
        loader_iter = tqdm(loader, desc=f'Epoch {epoch}') if is_master else loader
        for batch in loader_iter:
            noisy, clean = batch['pcl_noisy'], batch['pcl_clean']
            seeds, std = batch['seed_pnts'], batch['pcl_std']
            B, N = noisy.shape[0], noisy.shape[1]

            # ---- 1. backbone step, verbatim train_on_starter.py ----
            model.train()
            loss_b = model.get_supervised_loss(
                pcl_noisy=noisy, pcl_clean=clean,
                pcl_seeds=seeds, pcl_std=std,
                mask_size=args.mask_size,
                t_min=args.t_min, t_norm=args.t_norm,
            )
            opt_b.step(loss_b)
            b_losses.append(loss_b.item())

            # ---- 2. head step, verbatim train_refine.py -- against the
            #         backbone as it is RIGHT NOW ----
            if head_on:
                x = noisy - seeds.repeat(1, N, 1)
                y = clean - seeds.repeat(1, clean.shape[1], 1)

                # eval: the rollout must see the same BatchNorm behaviour
                # (running stats) inference will, and must not pollute those
                # stats with multi-step denoised inputs
                model.eval()
                # noise level is known during training; Eq. 15's estimate is
                # for inference, where it is not
                sig = float(std.mean().item()) * args.sigma_scale
                tau = max(sched.find_t_for_sigma(sig), args.diffusion_L)
                den = model.denoise_langevin_dynamics_diffusion(
                    x, L=args.diffusion_L, t_start=tau, t_norm=args.t_norm)
                with jt.no_grad():
                    _, feat = model._patch_forward(den, feat_T=None,
                                                   t_frac_val=0.0)
                    # nearest clean point, not same-index: the backbone
                    # scrambles points along the surface, and that scramble is
                    # unlearnable (see train_refine.py's header)
                    target = model.compute_gt_score(den, y)

                head.train()
                corr = head(den, feat)
                err = ((corr - target) ** 2).sum(dim=-1)          # (B, N)
                if use_mask:
                    m = np.zeros((B, N), dtype=np.float32)
                    m[:, :args.mask_size] = 1.0
                    loss_h = (err * jt.array(m)).sum() / float(B * args.mask_size)
                else:
                    loss_h = err.mean()

                if args.head_cd_weight > 0:
                    # Set-level CD on the refined CENTRAL region against the
                    # central clean points. Both sides are restricted to the
                    # same M rows (KNN columns are seed-distance ordered, and
                    # noisy/clean share row indices), so neither direction is
                    # polluted by the patch edge -- the trap that sank the
                    # backbone coverage term. This is the one place in the
                    # pipeline where a CD loss meets a module with a
                    # neighbourhood receptive field AND an inference path
                    # identical to its training path.
                    xr = (den + corr)[:, :M, :]
                    ym = y[:, :M, :]
                    d1, d2, _, _ = chamfer_dist(xr, ym)
                    cd = (jt.sqrt(jt.clamp(d1, min_v=1e-12)).mean()
                          + jt.sqrt(jt.clamp(d2, min_v=1e-12)).mean())
                    h_cds.append(float(cd.item()))
                    loss_h = loss_h + args.head_cd_weight * cd

                opt_h.step(loss_h)
                h_losses.append(loss_h.item())
                model.train()

            if is_master:
                desc = f'Epoch {epoch}, loss {np.mean(b_losses):.6f}'
                if h_losses:
                    desc += f', head {np.mean(h_losses):.6f}'
                if h_cds:
                    desc += f' (cd {np.mean(h_cds):.2e})'
                loader_iter.set_description(desc)

        if is_master:
            log_path = os.path.join(args.save_dir, 'joint_log.csv')
            write_header = not os.path.exists(log_path)
            with open(log_path, 'a') as f:
                if write_header:
                    f.write('epoch,backbone_loss,head_loss,head_cd\n')
                f.write(f'{epoch},{np.mean(b_losses):.6f},'
                        f'{np.mean(h_losses) if h_losses else float("nan"):.6f},'
                        f'{np.mean(h_cds) if h_cds else float("nan"):.8f}\n')

        if is_master and ((epoch + 1) % args.save_interval == 0
                          or epoch == args.epochs - 1):
            ckpt = os.path.join(args.save_dir, f'asdn-epoch{epoch:03d}.pkl')
            model.save(ckpt)
            print(f'Saved {ckpt}')
            if head_on:
                # only once it has actually trained: an identity head on disk
                # invites evaluating it by mistake
                hck = os.path.join(args.save_dir, f'refine-epoch{epoch:03d}.pkl')
                head.save(hck)
                print(f'Saved {hck}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ---- data: same knobs as train_on_starter.py ----
    parser.add_argument('--data_root', type=str, default='./dataset_train')
    parser.add_argument('--datalist', type=str, default='./datalist/train.txt')
    parser.add_argument('--num_samples', type=int, default=32768)
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--mesh_name', type=str, default='models/model_normalized.obj')
    parser.add_argument('--noise_dist', type=str, default='laplace',
                        choices=['laplace', 'gaussian'])
    parser.add_argument('--max_sigma', type=float, default=None,
                        help='must match predict_on_starter.py; raise for round B '
                             'if its noise exceeds ~3%%')
    # ---- backbone training ----
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=1e-5,
                        help='cosine floor for BOTH optimizers; pass a negative '
                             'value for constant lr')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--use_fusion', action='store_true')
    parser.add_argument('--mask_size', type=int, default=256)
    parser.add_argument('--t_min', type=int, default=30)
    parser.add_argument('--t_norm', type=str, default='T', choices=['T', 'tau'])
    parser.add_argument('--fusion_k', type=int, default=16)
    parser.add_argument('--fusion_gate', type=str, default='pos',
                        choices=['pos', 'posfeat'])
    parser.add_argument('--fusion_include_self', action='store_true')
    parser.add_argument('--static_depth', action='store_true',
                        help='required for multi-GPU; see train_on_starter.py')
    parser.add_argument('--no_bn_sync', action='store_true')
    parser.add_argument('--classify_ckpt', type=str, default=None)
    parser.add_argument('--classify_frame_knn', type=int, default=32)
    parser.add_argument('--init_ckpt', type=str, default=None)
    # ---- head training ----
    parser.add_argument('--head_start_epoch', type=int, default=20,
                        help='epoch at which the head starts training on the '
                             'evolving backbone. Earlier = more head steps but '
                             'noisier rollouts and slower early epochs.')
    parser.add_argument('--head_lr', type=float, default=1e-3,
                        help="train_refine.py's lr; cosine-decayed over the "
                             "head's own window")
    parser.add_argument('--head_cd_weight', type=float, default=0.0,
                        help='UNTESTED next experiment: add a symmetric CD loss '
                             'over the refined central region. The score loss '
                             'cannot see collapse (its target is not injective) '
                             'and three backbone-side regularisers failed to '
                             'give it eyes; this puts CD itself on the one '
                             'module whose training path equals its inference '
                             'path. Sweep against tune20 before trusting it. '
                             '0 keeps the production head loss exactly.')
    parser.add_argument('--diffusion_L', type=int, default=3,
                        help='must match inference')
    parser.add_argument('--sigma_scale', type=float, default=1.5,
                        help='must match inference -- and remember the lesson: '
                             'a retrained backbone needs this re-swept')
    parser.add_argument('--feat_dim', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--head_k', type=int, default=16)
    parser.add_argument('--head_layers', type=int, default=2)
    parser.add_argument('--init_head', type=str, default=None)
    # ---- bookkeeping ----
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='experiments/joint')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    if args.lr_min is not None and args.lr_min < 0:
        args.lr_min = None

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
