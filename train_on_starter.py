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

    model = DenoiseNetCD(classify_ckpt=args.classify_ckpt, classify_frame_knn=args.classify_frame_knn)
    if args.use_fusion:
        model.feature_nets.use_fusion = True
    
    if args.init_ckpt is not None:
        model.load(args.init_ckpt)
        if is_master:
            print(f'Loaded init weights from {args.init_ckpt}')
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
    )

    for epoch in range(args.epochs):
        model.train()
        losses = []
        loader_iter = tqdm(loader, desc=f'Epoch {epoch}') if is_master else loader
        for batch in loader_iter:
            # bridge collates dict-of-arrays into batched jt.Var
            loss = model.get_supervised_loss(
                pcl_noisy=batch['pcl_noisy'],
                pcl_clean=batch['pcl_clean'],
                pcl_seeds=batch['seed_pnts'],
                pcl_std=batch['pcl_std'],
            )
            optimizer.step(loss)  # Jittor auto all-reduces gradients across GPUs here
            losses.append(loss.item())
            if is_master:
                loader_iter.set_description(f'Epoch {epoch}, loss {np.mean(losses):.6f}')
            
        if is_master:
            log_path = os.path.join(args.save_dir, 'train_log.csv')
            write_header = not os.path.exists(log_path)
            with open(log_path, 'a') as f:
                if write_header:
                    f.write('epoch,loss\n')
                f.write(f'{epoch},{np.mean(losses):.6f}\n')

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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--use_fusion', action='store_true',
                        help='use the new FusionHead (Feature/Gradient Fusion) output head')
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