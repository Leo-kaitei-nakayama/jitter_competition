import os
import argparse
import shutil
import logging

import numpy as np
import jittor as jt
from tqdm.auto import tqdm

from models.denoiseCD import DenoiseNetCD
from models.utils import chamfer_distance_unit_sphere
from datasets.pcl import PointCloudDataset
from datasets.patch import PairedPatchDataset
from utils.misc import get_log_dir_name_tblogger, seed_all, str_list, get_logger, log_hyperparams
from utils.transforms import standard_train_transforms

jt.flags.use_cuda = 1


class ReduceLROnPlateau:
    """Minimal reimplementation of torch's ReduceLROnPlateau for jt.optim."""

    def __init__(self, optimizer, patience=2, factor=0.5, min_lr=1e-9):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best = float('inf')
        self.num_bad = 0

    def step(self, metric):
        if metric < self.best:
            self.best = metric
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad > self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                if new_lr < self.optimizer.lr:
                    print(f'[scheduler] reducing lr: {self.optimizer.lr:.2e} -> {new_lr:.2e}')
                    self.optimizer.lr = new_lr
                self.num_bad = 0


def build_train_loader(args):
    train_dset = PairedPatchDataset(
        datasets=[
            PointCloudDataset(
                root=args.dataset_root,
                dataset=args.dataset,
                split='train',
                resolution=resl,
                transform=standard_train_transforms(
                    noise_std_max=args.noise_max,
                    noise_std_min=args.noise_min,
                    rotate=args.aug_rotate)
            ) for resl in args.resolutions
        ],
        split='train',
        patch_size=args.patch_size,
        num_patches=args.patches_per_shape_per_epoch,
        patch_ratio=args.patch_ratio,
        on_the_fly=True,
    )
    # In Jittor, the Dataset IS the loader — configure batching here.
    train_dset.set_attrs(batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    return train_dset


def build_val_loader(args):
    val_dset = PointCloudDataset(
        root=args.dataset_root,
        dataset=args.dataset,
        split='test',
        resolution=args.resolutions[0],
        transform=standard_train_transforms(
            noise_std_max=args.val_noise, noise_std_min=args.val_noise, rotate=False),
    )
    val_dset.set_attrs(batch_size=args.val_batch_size, shuffle=False, num_workers=4)
    return val_dset


def validate(model, val_loader):
    model.eval()
    chamfers = []
    with jt.no_grad():
        for batch in val_loader:
            pcl_clean = batch['pcl_clean']
            pcl_noisy = batch['pcl_noisy']

            all_clean, all_denoised = [], []
            for i in range(pcl_noisy.shape[0]):
                pcl_denoised = model.patch_based_denoise(pcl_noisy[i], seed_k_alpha=10)
                all_clean.append(pcl_clean[i].unsqueeze(0))
                all_denoised.append(pcl_denoised.unsqueeze(0))

            all_clean = jt.concat(all_clean, dim=0)
            all_denoised = jt.concat(all_denoised, dim=0)
            avg_chamfer = chamfer_distance_unit_sphere(all_denoised, all_clean,
                                                        batch_reduction='mean')[0].item()
            chamfers.append(avg_chamfer)
    model.train()
    return float(np.mean(chamfers)) if chamfers else float('inf')


def main(args):
    # Logging
    log_dir_name = get_log_dir_name_tblogger(name='D%s_' % (args.dataset))
    log_dir = os.path.join(args.log_root, log_dir_name)
    if args.resume_from_checkpoint is not None:
        log_dir = os.path.dirname(args.resume_from_checkpoint)
        log_dir_name = os.path.basename(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.resume_from_checkpoint is None:
        files_to_save = ['./models/feature.py', './models/blocks.py',
                         './models/utils.py', './models/denoiseCD.py']
        for file_ in files_to_save:
            if os.path.exists(file_):
                shutil.copyfile(file_, os.path.join(log_dir, os.path.basename(file_)))

    logger = get_logger('train_ASDN', log_dir)
    log_hyperparams(log_dir, args)

    for k, v in vars(args).items():
        logger.info('[ARGS::%s] %s' % (k, repr(v)))

    # Model
    logger.info('INFO: Building model...')
    model = DenoiseNetCD(args)
    if args.resume_from_checkpoint is not None:
        model.load(args.resume_from_checkpoint)
        logger.info(f'INFO: Resumed from {args.resume_from_checkpoint}')

    optimizer = jt.optim.Adam(model.feature_nets.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=args.sched_patience,
                                   factor=args.sched_factor, min_lr=args.min_lr)

    train_loader = build_train_loader(args)
    val_loader = build_val_loader(args)

    logger.info('INFO: Start training...')
    try:
        for epoch in range(args.max_epochs):
            model.train()
            train_losses = []
            for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
                pcl_noisy = batch['pcl_noisy']
                pcl_clean = batch['pcl_clean']
                pcl_seeds = batch['seed_pnts']
                pcl_std = batch['pcl_std']

                loss = model.get_supervised_loss(
                    pcl_noisy=pcl_noisy, pcl_clean=pcl_clean,
                    pcl_seeds=pcl_seeds, pcl_std=pcl_std)

                optimizer.step(loss)  # Jittor: backward + update in one call
                train_losses.append(loss.item())

            logger.info(f'INFO: Current epoch training loss: {np.mean(train_losses):.6f}')

            # Validation + checkpointing every save_interval epochs
            if (epoch + 1) % args.save_interval == 0:
                val_loss = validate(model, val_loader)
                logger.info(f'INFO: Current epoch validation loss: {val_loss:.6f}')
                scheduler.step(val_loss)

                ckpt_path = os.path.join(
                    log_dir, f'denoisenet-epoch{epoch:02d}-val_loss{val_loss:.8f}.pkl')
                model.save(ckpt_path)
                logger.info(f'INFO: Saved checkpoint to {ckpt_path}')

    except KeyboardInterrupt:
        logger.info('INFO: Terminating...')
        print('Terminating...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='PUNet')
    parser.add_argument('--changelog', type=str, default='')
    parser.add_argument('--patches_per_shape_per_epoch', type=int, default=1000)
    parser.add_argument('--patch_ratio', type=float, default=1.2)
    parser.add_argument('--resolutions', type=str_list,
                        default=['10000_poisson', '30000_poisson', '50000_poisson'])
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--train_batch_size', type=int, default=24)
    parser.add_argument('--val_batch_size', type=int, default=24)
    parser.add_argument('--noise_lvs', type=list, default=None)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])

    ## Optimizer and scheduler
    parser.add_argument('--sched_patience', default=2, type=int)
    parser.add_argument('--sched_factor', default=0.5, type=float)
    parser.add_argument('--min_lr', default=1e-9, type=float)
    parser.add_argument('--lr', type=float, default=5e-4)

    ## Training
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--log_root', type=str, default='./logs/ASDN')
    parser.add_argument('--val_noise', type=float, default=0.015)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--max_epochs', type=int, default=800)

    # Ablation parameters
    parser.add_argument('--patch_size', type=int, default=1000)

    args = parser.parse_args()

    seed_all(args.seed)
    main(args)
