"""
Run ASDN denoising on the starter competition's test set and write the
submission format.

Input:   <data_root>/shapenet/<synset>/<model_id>/noisy.npy   (N,3)
Output:  <out_root>/shapenet/<synset>/<model_id>/denoised.npy  (N,3) float32

Usage:
    python predict_on_starter.py \
        --ckpt experiments/asdn/asdn-epoch099.pkl \
        --data_root ./dataset_test_noisy \
        --datalist ./datalist/test.txt \
        --out_root ./results/dataset_test_noisy

Then zip for submission:
    cd results/dataset_test_noisy && zip -r ../../result.zip shapenet/
"""
import os
import argparse
import numpy as np
import jittor as jt
from tqdm import tqdm

from models.denoiseCD import DenoiseNetCD
from models.refine import RefineHead
from bridge.data_bridge import ShapeNetNoisyPredictDataset, normalize_unit_sphere

jt.flags.use_cuda = 1


def load_model(args):
    """Loads a Jittor .pkl checkpoint produced by train_on_starter.py.
    (No torch dependency, no external-data checkpoints -- everything here
    was trained on the competition's own data only.)

    The fusion_* options must match the ones the checkpoint was trained with,
    otherwise the fusion head's weights will not line up."""
    model = DenoiseNetCD(fusion_k=args.fusion_k, fusion_gate=args.fusion_gate,
                         fusion_include_self=args.fusion_include_self,
                         static_depth=args.static_depth)
    if args.use_fusion:
        model.feature_nets.use_fusion = True
    model.load(args.ckpt)
    model.eval()

    head = None
    if args.refine_ckpt is not None:
        head = RefineHead(feat_dim=args.feat_dim, hidden=args.hidden,
                          k=args.head_k, n_layers=args.head_layers)
        head.load(args.refine_ckpt)
        head.eval()
        print(f'refine head: {args.refine_ckpt}')
    return model, head


def main(args):
    model, refine_head = load_model(args)
    model.set_predict(True) if hasattr(model, 'set_predict') else None
    model.eval()

    ds = ShapeNetNoisyPredictDataset(
        root=args.data_root,
        datalist=args.datalist,
        data_name=args.data_name,
        batch_size=1,
        num_workers=args.num_workers,
    )

    taus = []
    for batch in tqdm(ds, desc='Predicting'):
        # batch_size=1; unwrap
        pc_noisy = batch['pc_noisy']
        rel = batch['rel']
        if isinstance(rel, (list, tuple)):
            rel = rel[0]
        if isinstance(pc_noisy, jt.Var):
            pc_noisy_np = pc_noisy.numpy()
        else:
            pc_noisy_np = np.asarray(pc_noisy)
        pc_noisy_np = pc_noisy_np.reshape(-1, 3).astype(np.float32)

        # Normalize to unit sphere (record center/scale to invert afterwards)
        pc_norm, center, scale = normalize_unit_sphere(pc_noisy_np)

        with jt.no_grad():
            pcl = jt.array(pc_norm)
            if args.use_diffusion:
                pcl, tau = model.patch_based_denoise_diffusion(
                    pcl_noisy=pcl,
                    patch_size=args.patch_size,
                    seed_k=args.seed_k,
                    seed_k_alpha=args.seed_k_alpha,
                    L=args.diffusion_L,
                    t_start=args.diffusion_t_start,
                    adaptive=not args.fixed_schedule,
                    sigma_scale=args.sigma_scale,
                    sigma_estimator=args.sigma_estimator,
                    t_norm=args.t_norm,
                    return_tau=True,
                    refine_head=refine_head,
                )
                taus.append((rel, tau))
            else:
                for _ in range(args.niters):
                    pcl = model.patch_based_denoise(
                        pcl_noisy=pcl,
                        patch_size=args.patch_size,
                        seed_k=args.seed_k,
                        seed_k_alpha=args.seed_k_alpha,
                    )
            denoised = pcl.numpy().astype(np.float32)

        # Denormalize back to the original coordinate frame
        denoised = denoised * scale + center

        out_dir = os.path.join(args.out_root, rel)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, args.out_name), denoised.astype(np.float32))

    if taus:
        # sigma_bar is linear in t for this schedule (sigma_bar[632] = 0.02), so
        # sigma ~= 3.16e-5 * tau. Quick sanity read on whether the adaptive
        # schedule is tracking the data: these should land inside the noise range
        # the model was trained on.
        tv = np.array([t for _, t in taus], dtype=np.float64)
        print(f'[adaptive schedule] tau over {len(tv)} clouds: '
              f'min={tv.min():.0f} mean={tv.mean():.0f} max={tv.max():.0f} '
              f'(sigma ~ {3.16e-5 * tv.min():.4f} .. {3.16e-5 * tv.max():.4f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Jittor .pkl (from train_on_starter.py) or original torch .ckpt')
    parser.add_argument('--data_root', type=str, default='./dataset_test_noisy')
    parser.add_argument('--datalist', type=str, default='./datalist/test.txt')
    parser.add_argument('--data_name', type=str, default='noisy.npy')
    parser.add_argument('--out_root', type=str, default='./results/dataset_test_noisy')
    parser.add_argument('--out_name', type=str, default='denoised.npy')
    parser.add_argument('--use_fusion', action='store_true',
                        help='enable the FusionHead output head (must match training)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_diffusion', action='store_true',
                        help='use L-step diffusion sampling (Alg. 2) instead of niters loop')
    parser.add_argument('--diffusion_L', type=int, default=5)
    parser.add_argument('--diffusion_t_start', type=int, default=632,
                        help='starting timestep when --fixed_schedule is set; also the '
                             'probe timestep for the adaptive estimate when --t_norm=T')
    parser.add_argument('--fixed_schedule', action='store_true',
                        help='disable the adaptive schedule (Eq. 15/16) and start every '
                             'cloud at --diffusion_t_start. This is the paper\'s '
                             '"FixedSched" baseline; adaptive is on by default.')
    parser.add_argument('--sigma_scale', type=float, default=1.5,
                        help='multiplier on the estimated noise sigma before picking tau. '
                             'The Eq. 15 estimator reads low, so the model is told the '
                             'cloud is cleaner than it is and under-denoises. 1.5 was '
                             'measured as the optimum on a 30-sample local eval set '
                             '(76.94 vs 74.30 at 1.0; CD and P2S both peak there, so it '
                             'is a calibration fix, not a trade-off). Re-sweep it against '
                             'your own eval set after retraining -- it is model-specific '
                             'and the cheapest single knob for score.')
    parser.add_argument('--sigma_estimator', type=str, default='rms', choices=['var', 'rms'],
                        help="'var' is Eq. 15 literally, but Var(||s||) underestimates sigma "
                             'by ~1.66x for isotropic noise, so it systematically '
                             "under-denoises. 'rms' = sqrt(mean(||s||^2)) recovers the true "
                             'timestep exactly in that case and is the default here.')
    parser.add_argument('--t_norm', type=str, default='T', choices=['T', 'tau'],
                        help='must match the value train_on_starter.py was run with')
    parser.add_argument('--fusion_k', type=int, default=16,
                        help='must match training (paper uses 32)')
    parser.add_argument('--fusion_gate', type=str, default='pos', choices=['pos', 'posfeat'],
                        help='must match training')
    parser.add_argument('--fusion_include_self', action='store_true',
                        help='must match training')
    parser.add_argument('--static_depth', action='store_true',
                        help='must match training')
    parser.add_argument('--refine_ckpt', type=str, default=None,
                        help='trained RefineHead from train_refine.py, applied per '
                             'patch after the diffusion steps. Requires --use_diffusion.')
    parser.add_argument('--feat_dim', type=int, default=32,
                        help='refine head: encoder feature width, must match training')
    parser.add_argument('--hidden', type=int, default=64,
                        help='refine head: hidden width, must match training')
    parser.add_argument('--head_k', type=int, default=16,
                        help='refine head: neighbours, must match training')
    parser.add_argument('--head_layers', type=int, default=2,
                        help='refine head: EdgeConv layers, must match training')

    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--niters', type=int, default=1)
    parser.add_argument('--seed_k', type=int, default=6)
    parser.add_argument('--seed_k_alpha', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    main(args)