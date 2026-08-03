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
from bridge.data_bridge import ShapeNetNoisyPredictDataset, normalize_unit_sphere

jt.flags.use_cuda = 1


def load_model(ckpt_path, use_fusion):
    """Loads a Jittor .pkl checkpoint produced by train_on_starter.py.
    (No torch dependency, no external-data checkpoints -- everything here
    was trained on the competition's own data only.)"""
    model = DenoiseNetCD()
    if use_fusion:
        model.feature_nets.use_fusion = True
    model.load(ckpt_path)
    model.eval()
    return model


def main(args):
    model = load_model(args.ckpt, args.use_fusion)
    model.set_predict(True) if hasattr(model, 'set_predict') else None
    model.eval()

    ds = ShapeNetNoisyPredictDataset(
        root=args.data_root,
        datalist=args.datalist,
        data_name=args.data_name,
        batch_size=1,
        num_workers=args.num_workers,
    )

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
                pcl = model.patch_based_denoise_diffusion(
                    pcl_noisy=pcl,
                    patch_size=args.patch_size,
                    seed_k=args.seed_k,
                    seed_k_alpha=args.seed_k_alpha,
                    L=args.diffusion_L,
                    t_start=args.diffusion_t_start,
                )
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
    parser.add_argument('--diffusion_t_start', type=int, default=632)

    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--niters', type=int, default=1)
    parser.add_argument('--seed_k', type=int, default=6)
    parser.add_argument('--seed_k_alpha', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    main(args)