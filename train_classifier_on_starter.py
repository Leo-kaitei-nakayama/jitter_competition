"""
Train the Classify (ScaleNet) submodule from scratch, using ONLY the
competition's ShapeNet training data.

Why this exists: models/feature.py used to auto-load the original ASDN
paper's pretrained/classify.ckpt. That checkpoint was trained on ASDN's own
(non-competition) dataset, which violates this competition's "no outside
data" rule. So classify must be trained here, from scratch, on the provided
ShapeNet data only -- producing a plain Jittor .pkl checkpoint with no
torch dependency anywhere.

Usage:
    python train_classifier_on_starter.py \
        --data_root ./dataset_train \
        --datalist ./datalist/train.txt \
        --epochs 100 --batch_size 8 \
        --save_dir experiments/classify

Then pass the resulting .pkl to train_on_starter.py's --classify_ckpt so the
denoiser's FeatureExtraction uses it instead of building an untrained one.
"""
import os
import argparse
import numpy as np
import jittor as jt
from tqdm import tqdm

from models.classify import Classify
from bridge.data_bridge import ShapeNetPatchTrainDataset

jt.flags.use_cuda = 1


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    model = Classify(frame_knn=args.frame_knn)
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
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            loss = model.get_supervised_loss_nn(
                pcl_noisy=batch['pcl_noisy'],
                pcl_clean=batch['pcl_clean'],
                pcl_seeds=batch['seed_pnts'],
                pcl_std=batch['pcl_std'],
            )
            optimizer.step(loss)
            losses.append(loss.item())
            pbar.set_description(f'Epoch {epoch}, loss {np.mean(losses):.6f}')

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            ckpt = os.path.join(args.save_dir, f'classify-epoch{epoch:03d}.pkl')
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
    parser.add_argument('--frame_knn', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='experiments/classify')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    main(args)
