"""
Train a RefineHead on top of a frozen denoiser.

The backbone never changes. Every forward pass through it runs under
jt.no_grad(), and only the head is handed to the optimizer, so no gradient can
reach a trained weight -- the same guarantee postprocess.py gets from being a
separate process, one level deeper in.

Each step:

    noisy patch
        -> [frozen model, L diffusion steps]  -> denoised patch
        -> [frozen encoder, one more pass]    -> E(x) at the denoised positions
        -> [RefineHead]                       -> corrective displacement
        -> loss against NN(denoised, clean) - denoised

The target is the displacement to the NEAREST clean point, not to the
same-index one. The model moves points along the surface, so output point i is
near the surface but generally not near clean point i; that tangential
scrambling is arbitrary and unlearnable, while the distance to the surface is
what check_learnable.py measured as partly predictable (autocorrelation +0.35).

The frozen pass uses the timestep implied by the patch's known noise level
rather than estimating it, since during training the noise level is not a
secret. Everything else matches inference.

Usage:
    mpirun -np 6 python -u train_refine.py \
        --ckpt experiments/asdn_diffusion/asdn-epoch049.pkl \
        --use_fusion --static_depth \
        --epochs 20 --batch_size 48 --num_workers 2 --lr 1e-3 \
        --save_dir experiments/refine

Then pass the result to predict_on_starter.py via --refine_ckpt.
"""
import os
import argparse

import numpy as np
import jittor as jt
from tqdm import tqdm

from models.denoiseCD import DenoiseNetCD
from models.refine import RefineHead
from bridge.data_bridge import ShapeNetPatchTrainDataset

jt.flags.use_cuda = 1


def main(args):
    is_master = (jt.rank == 0)
    if is_master:
        os.makedirs(args.save_dir, exist_ok=True)

    # ---- frozen backbone ----
    model = DenoiseNetCD(fusion_k=args.fusion_k, fusion_gate=args.fusion_gate,
                         fusion_include_self=args.fusion_include_self,
                         static_depth=args.static_depth,
                         max_sigma=args.max_sigma)
    if args.use_fusion:
        model.feature_nets.use_fusion = True
    model.load(args.ckpt)
    model.eval()
    for p in model.parameters():
        p.stop_grad()
    if is_master:
        print(f'frozen backbone: {args.ckpt}')

    # ---- trainable head ----
    head = RefineHead(feat_dim=args.feat_dim, hidden=args.hidden,
                      k=args.head_k, n_layers=args.head_layers)
    if args.init_head is not None:
        head.load(args.init_head)
    head.train()
    optimizer = jt.optim.Adam(head.parameters(), lr=args.lr)

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

    for epoch in range(args.epochs):
        head.train()
        losses = []
        it = tqdm(loader, desc=f'Epoch {epoch}') if is_master else loader

        for batch in it:
            noisy, clean = batch['pcl_noisy'], batch['pcl_clean']
            seeds, std = batch['seed_pnts'], batch['pcl_std']
            B, N = noisy.shape[0], noisy.shape[1]

            x = noisy - seeds.repeat(1, N, 1)
            y = clean - seeds.repeat(1, clean.shape[1], 1)

            # ---- frozen: denoise the patch exactly as inference would ----
            # the noise level is known here, so tau comes from it directly
            # instead of Eq. 15's estimate
            sig = float(std.mean().item()) * args.sigma_scale
            tau = max(sched.find_t_for_sigma(sig), args.diffusion_L)
            with jt.no_grad():
                den = model.denoise_langevin_dynamics_diffusion(
                    x, L=args.diffusion_L, t_start=tau, t_norm=args.t_norm)
                _, feat = model._patch_forward(den, feat_T=None, t_frac_val=0.0)

            # ---- target: displacement to the nearest clean point ----
            with jt.no_grad():
                target = model.compute_gt_score(den, y)

            # ---- trainable ----
            corr = head(den, feat)
            err = ((corr - target) ** 2).sum(dim=-1)          # (B, N)
            if use_mask:
                m = np.zeros((B, N), dtype=np.float32)
                m[:, :args.mask_size] = 1.0
                err = err * jt.array(m)
                loss = err.sum() / float(B * args.mask_size)
            else:
                loss = err.mean()

            optimizer.step(loss)
            losses.append(loss.item())
            if is_master:
                it.set_description(f'Epoch {epoch}, loss {np.mean(losses):.8f}')

        if is_master:
            log = os.path.join(args.save_dir, 'refine_log.csv')
            new = not os.path.exists(log)
            with open(log, 'a') as f:
                if new:
                    f.write('epoch,loss\n')
                f.write(f'{epoch},{np.mean(losses):.8f}\n')

            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                out = os.path.join(args.save_dir, f'refine-epoch{epoch:03d}.pkl')
                head.save(out)
                print(f'Saved {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True,
                   help='frozen denoiser checkpoint')
    p.add_argument('--data_root', type=str, default='./dataset_train')
    p.add_argument('--datalist', type=str, default='./datalist/train.txt')
    p.add_argument('--num_samples', type=int, default=32768)
    p.add_argument('--patch_size', type=int, default=1000)
    p.add_argument('--noise_min', type=float, default=0.005)
    p.add_argument('--noise_max', type=float, default=0.02)
    p.add_argument('--mesh_name', type=str, default='models/model_normalized.obj')
    p.add_argument('--max_sigma', type=float, default=None,
                   help='must match the frozen checkpoint and inference')
    p.add_argument('--noise_dist', type=str, default='laplace',
                   choices=['laplace', 'gaussian'])
    p.add_argument('--batch_size', type=int, default=48)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--mask_size', type=int, default=256)
    p.add_argument('--save_interval', type=int, default=2)
    p.add_argument('--save_dir', type=str, default='experiments/refine')
    p.add_argument('--seed', type=int, default=2024)
    # must match the frozen checkpoint
    p.add_argument('--use_fusion', action='store_true')
    p.add_argument('--static_depth', action='store_true')
    p.add_argument('--fusion_k', type=int, default=16)
    p.add_argument('--fusion_gate', type=str, default='pos', choices=['pos', 'posfeat'])
    p.add_argument('--fusion_include_self', action='store_true')
    p.add_argument('--t_norm', type=str, default='T', choices=['T', 'tau'])
    # must match how you will run inference
    p.add_argument('--diffusion_L', type=int, default=3)
    p.add_argument('--sigma_scale', type=float, default=1.5)
    # head shape
    p.add_argument('--feat_dim', type=int, default=32,
                   help="width of the encoder feature; 32 for this architecture")
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--head_k', type=int, default=16)
    p.add_argument('--head_layers', type=int, default=2)
    p.add_argument('--init_head', type=str, default=None)
    args = p.parse_args()

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
