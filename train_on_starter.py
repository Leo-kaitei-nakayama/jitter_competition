"""
Train the ASDN denoiser on the starter competition's ShapeNet training set.

Usage:
    python train_on_starter.py \
        --data_root ./dataset_train \
        --datalist ./datalist/train.txt \
        --epochs 100 --batch_size 8 --save_dir experiments/asdn

Produces experiments/asdn/asdn-epochXX.pkl checkpoints; pass the best one to
predict_on_starter.py.
"""
import os
import argparse
import numpy as np
import jittor as jt
from tqdm import tqdm

from models.denoiseCD import DenoiseNetCD
from bridge.data_bridge import ShapeNetPatchTrainDataset

jt.flags.use_cuda = 1


def main(args):
    is_master = (jt.rank == 0)  # only rank 0 logs / saves, avoids 8x duplicate writes
    if is_master:
        os.makedirs(args.save_dir, exist_ok=True)

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
            # model.load() just replaced every submodule's weights with the ones
            # embedded in init_ckpt -- including the classifier, which in a
            # checkpoint trained under --static_depth is still random init. The
            # explicitly requested classifier must win, so re-apply it.
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
    model.train()
    optimizer = jt.optim.Adam(model.feature_nets.parameters(), lr=args.lr)

    loader = ShapeNetPatchTrainDataset(
        root=args.data_root,
        datalist=args.datalist,
        num_samples=args.num_samples,
        patch_size=args.patch_size,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        noise_dist=args.noise_dist,
        mesh_name=args.mesh_name,
        with_normals=(args.tangent_weight > 0),
    )

    for epoch in range(args.epochs):
        if args.lr_min is not None:
            # cosine decay from --lr to --lr_min across the run. The epoch-60
            # loss curve wobbles in a 0.00088-0.00096 band under constant lr,
            # which is the signature of being step-size-limited, not converged.
            import math as _math
            optimizer.lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
                1 + _math.cos(_math.pi * epoch / max(1, args.epochs - 1)))
            if is_master:
                print(f'epoch {epoch}: lr = {optimizer.lr:.2e}')
        model.train()
        losses, score_losses, unif_raws, cov_raws, tang_raws = [], [], [], [], []
        loader_iter = tqdm(loader, desc=f'Epoch {epoch}') if is_master else loader
        for batch in loader_iter:
            # bridge collates dict-of-arrays into batched jt.Var
            loss = model.get_supervised_loss(
                pcl_noisy=batch['pcl_noisy'],
                pcl_clean=batch['pcl_clean'],
                pcl_seeds=batch['seed_pnts'],
                pcl_std=batch['pcl_std'],
                mask_size=args.mask_size,
                t_min=args.t_min,
                t_norm=args.t_norm,
                exact_score=args.exact_score,
                unif_weight=args.uniformity_weight,
                cov_weight=args.coverage_weight,
                tang_weight=args.tangent_weight,
                pcl_normals=batch.get('pcl_normals'),
            )
            optimizer.step(loss)  # Jittor auto all-reduces gradients across GPUs here
            losses.append(loss.item())
            score_losses.append(model.last_score_loss)
            unif_raws.append(model.last_unif_raw)
            cov_raws.append(model.last_cov_raw)
            tang_raws.append(model.last_tang_raw)
            if is_master:
                desc = f'Epoch {epoch}, loss {np.mean(losses):.6f}'
                if args.uniformity_weight > 0:
                    # what the uniformity term is actually worth: its share of the
                    # total, and the raw spacing variance it is driving down
                    share = args.uniformity_weight * np.mean(unif_raws) / max(np.mean(losses), 1e-12)
                    desc += f' (unif {np.mean(unif_raws):.4f}, {100*share:.1f}% of loss)'
                if args.coverage_weight > 0:
                    share = args.coverage_weight * np.mean(cov_raws) / max(np.mean(losses), 1e-12)
                    desc += f' (cov {np.mean(cov_raws):.2e}, {100*share:.1f}% of loss)'
                if args.tangent_weight > 0:
                    share = args.tangent_weight * np.mean(tang_raws) / max(np.mean(losses), 1e-12)
                    desc += f' (tang {np.mean(tang_raws):.2e}, {100*share:.1f}% of loss)'
                loader_iter.set_description(desc)
            
        if is_master:
            log_path = os.path.join(args.save_dir, 'train_log.csv')
            write_header = not os.path.exists(log_path)
            with open(log_path, 'a') as f:
                if write_header:
                    f.write('epoch,loss,score_loss,unif_raw,cov_raw,tang_raw\n')
                f.write(f'{epoch},{np.mean(losses):.6f},'
                        f'{np.mean(score_losses):.6f},{np.mean(unif_raws):.6f},'
                        f'{np.mean(cov_raws):.8f},{np.mean(tang_raws):.8f}\n')

        if is_master and ((epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1):
            ckpt = os.path.join(args.save_dir, f'asdn-epoch{epoch:03d}.pkl')
            model.save(ckpt)
            print(f'Saved {ckpt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset_train')
    parser.add_argument('--datalist', type=str, default='./datalist/train.txt')
    parser.add_argument('--num_samples', type=int, default=32768)
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--mesh_name', type=str, default='models/model_normalized.obj',
                        help='path to the mesh inside each datalist entry. Change this '
                             'when a new dataset lays its files out differently.')
    parser.add_argument('--max_sigma', type=float, default=None,
                        help='largest noise level the diffusion schedule can represent '
                             '(default 0.0316). Raise it for a dataset noisier than ~3%%, '
                             'or every loud cloud gets clamped to the same timestep. '
                             'predict_on_starter.py must be given the same value.')
    parser.add_argument('--noise_dist', type=str, default='laplace',
                        choices=['laplace', 'gaussian'],
                        help='additive noise shape. Either way noise_min/max are '
                             'standard deviations, matching the competition spec.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_min', type=float, default=None,
                        help='enable cosine lr decay from --lr down to this value '
                             'over --epochs. Recommended for continuing from a '
                             'checkpoint that plateaued under constant lr.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--use_fusion', action='store_true',
                        help='use the new FusionHead (Feature/Gradient Fusion) output head')
    # --- diffusion training strategy (paper Algorithm 1) ---
    parser.add_argument('--mask_size', type=int, default=256,
                        help='K_p: how many of the patch\'s innermost points carry the '
                             'loss. Stage 2 advances the rest with the GT score. 0 disables.')
    parser.add_argument('--t_min', type=int, default=30,
                        help='floor on sampled timesteps; the Eq. 9 weight diverges as t->0. '
                             'Must stay below the smallest t inference visits (~32 for the '
                             'quietest clouds at L=5).')
    parser.add_argument('--t_norm', type=str, default='T', choices=['T', 'tau'],
                        help="what the relative timestep is divided by. 'T' makes t_frac an "
                             'absolute noise-level signal and is what makes the adaptive '
                             "schedule bite at inference; 'tau' is the paper's convention "
                             'but makes t_frac identical for every cloud, which cancels the '
                             'adaptive estimate. Keep T unless you know why you want tau. '
                             'predict_on_starter.py must be given the same value.')
    parser.add_argument('--fusion_k', type=int, default=16,
                        help='neighbours per point in the gradient prediction/fusion '
                             'modules (paper uses 32)')
    parser.add_argument('--fusion_gate', type=str, default='pos', choices=['pos', 'posfeat'],
                        help="how FeatureFusion builds its gates. 'posfeat' conditions "
                             'them on E(x^t)/E(x^T) as the paper describes; '
                             "'pos' is the older position-only variant.")
    parser.add_argument('--fusion_include_self', action='store_true',
                        help='let each point be its own gradient-prediction neighbour')
    parser.add_argument('--uniformity_weight', type=float, default=0.0,
                        help='penalise uneven point spacing at x + score_hat, i.e. '
                             'where the network sends each point. The score target '
                             'NN(x, clean) - x is not injective, so the per-point '
                             'loss is fully satisfied when several points collapse '
                             'onto one clean point -- invisible to the loss, '
                             'punished by CD. This puts the missing signal in the '
                             'backbone, where the scramble originates, instead of '
                             'only in the downstream refine head. Sweep 1e-5..1e-3; '
                             '0 reproduces the original loss exactly.')
    parser.add_argument('--tangent_weight', type=float, default=0.0,
                        help='penalise the TANGENTIAL component of the predicted '
                             'displacement (E_disp from Xu/Yang/Deng 2024): '
                             '||n x score||^2 with n the true face normal at the '
                             'clean point the score aims at. Unlike the uniformity '
                             'penalty, a cross product leaves normal-direction motion '
                             'completely free, so it cannot buy distribution by paying '
                             'P2S. Loads exact mesh normals automatically. Sweep '
                             '0.1..3; 0 disables.')
    parser.add_argument('--coverage_weight', type=float, default=0.0,
                        help="weight of CD's second (coverage) term at x + score_hat: "
                             'every central clean point must have a sent point nearby. '
                             'The score loss already is CD\'s first term, so this '
                             'completes the metric inside the training loss. Same units '
                             'as a squared distance with a ~6e-6 floor, so it needs a '
                             'large weight: sweep 3..30. 0 disables.')
    parser.add_argument('--exact_score', action='store_true',
                        help='train against the exact per-point displacement '
                             '(pcl_clean - pcl) instead of Eq. 14\'s nearest-neighbour '
                             'approximation. Valid because data_bridge pairs the noisy '
                             'and clean patches row-by-row. Training-only -- '
                             'predict_on_starter.py needs no matching flag.')
    parser.add_argument('--static_depth', action='store_true',
                        help='run every sample through all 4 encoder/decoder layers '
                             'instead of letting the classifier pick a per-sample depth. '
                             'REQUIRED for multi-GPU: Jittor BatchNorm is SyncBN under '
                             'MPI, so a data-dependent block count makes ranks issue '
                             'different numbers of collectives and NCCL deadlocks. '
                             'predict_on_starter.py must be given the same value. '
                             'To train WITH adaptive depth under MPI, drop this '
                             'flag and pass --no_bn_sync instead.')
    parser.add_argument('--no_bn_sync', action='store_true',
                        help='disable BatchNorm cross-GPU synchronization. This is '
                             'the other fix for the multi-GPU deadlock: instead of '
                             'pinning the depth (--static_depth), remove the '
                             'collectives from the forward pass, so a data-dependent '
                             'depth cannot desynchronize the ranks. Each rank then '
                             'normalizes over its own batch_size/n_ranks samples, '
                             'which is fine at >=8 per rank. Required whenever '
                             'adaptive depth (a trained --classify_ckpt without '
                             '--static_depth) is trained under mpirun.')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--classify_ckpt', type=str, default=None,
                        help='Jittor .pkl from train_classifier_on_starter.py '
                             '(trained on competition data only). If omitted, '
                             'an untrained Classify is used.')
    parser.add_argument('--init_ckpt', type=str, default=None,
                        help='load U-Net weights from this checkpoint before training (fusion head keeps random init)')
    parser.add_argument('--classify_frame_knn', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='experiments/asdn')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    main(args)