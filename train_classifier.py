import os
import argparse

import numpy as np
import jittor as jt
from tqdm.auto import tqdm

from models.classify import Classify
from datasets.pcl import PointCloudDataset
from datasets.patch import PairedPatchDataset
from utils.misc import get_log_dir_name_tblogger, seed_all, str_list, get_logger, log_hyperparams
from utils.transforms import standard_train_transforms

jt.flags.use_cuda = 1


class ReduceLROnPlateau:
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
                    noise_std_max=args.noise_max, noise_std_min=args.noise_min,
                    rotate=args.aug_rotate)
            ) for resl in args.resolutions
        ],
        split='train',
        patch_size=args.patch_size,
        num_patches=args.patches_per_shape_per_epoch,
        patch_ratio=args.patch_ratio,
        transform=None,
    )
    train_dset.set_attrs(batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    return train_dset


def build_val_loader(args):
    val_dset = PointCloudDataset(
        root=args.dataset_root,
        dataset=args.dataset,
        split='test',
        resolution=args.resolutions[2] if len(args.resolutions) > 2 else args.resolutions[-1],
        transform=standard_train_transforms(
            noise_std_max=args.val_noise, noise_std_min=args.val_noise,
            rotate=False, scale_d=0.0),
    )
    val_dset.set_attrs(batch_size=args.val_batch_size, shuffle=False, num_workers=4)
    return val_dset


def validate(model, val_loader):
    model.eval()
    avgs = []
    with jt.no_grad():
        for batch in val_loader:
            pcl_clean = batch['pcl_clean']
            pcl_noisy = batch['pcl_noisy']

            all_clean, all_pred = [], []
            for i in range(pcl_noisy.shape[0]):
                pre_scale, scale_gt = model.patch_based_shang(pcl_noisy[i], pcl_clean[i])
                all_clean.append(scale_gt.unsqueeze(0))
                all_pred.append(pre_scale.unsqueeze(0))

            avg = jt.abs(jt.concat(all_clean, dim=0).squeeze(-1) -
                          jt.concat(all_pred, dim=0).squeeze(-1)).mean().item()
            avgs.append(avg)
    model.train()
    return float(np.mean(avgs)) if avgs else float('inf')


def main(args):
    log_dir_name = get_log_dir_name_tblogger(name='C%s_' % (args.dataset))
    log_dir = os.path.join(args.log_root, log_dir_name)
    os.makedirs(log_dir, exist_ok=True)

    logger = get_logger('train_classifier', log_dir)
    log_hyperparams(log_dir, args)
    for k, v in vars(args).items():
        logger.info('[ARGS::%s] %s' % (k, repr(v)))

    logger.info('INFO: Building model...')
    model = Classify(args)

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
                loss = model.get_supervised_loss_nn(
                    pcl_noisy=batch['pcl_noisy'],
                    pcl_clean=batch['pcl_clean'],
                    pcl_seeds=batch['seed_pnts'],
                    pcl_std=batch['pcl_std'])
                optimizer.step(loss)
                train_losses.append(loss.item())

            logger.info(f'INFO: Current epoch training loss: {np.mean(train_losses):.6f}')

            if (epoch + 1) % args.save_interval == 0:
                val_loss = validate(model, val_loader)
                logger.info(f'INFO: Current epoch validation loss: {val_loss:.6f}')
                scheduler.step(val_loss)

                ckpt_path = os.path.join(
                    log_dir, f'classify-epoch{epoch:02d}-val_loss{val_loss:.8f}.pkl')
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
    parser.add_argument('--patches_per_shape_per_epoch', type=int, default=1000)
    parser.add_argument('--patch_ratio', type=float, default=1.2)
    parser.add_argument('--resolutions', type=str_list,
                        default=['10000_poisson', '30000_poisson', '50000_poisson'])
    parser.add_argument('--noise_min', type=float, default=0.005)
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--train_batch_size', type=int, default=24)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])

    ## Model
    parser.add_argument('--frame_knn', type=int, default=32)

    ## Optimizer and scheduler
    parser.add_argument('--sched_patience', default=2, type=int)
    parser.add_argument('--sched_factor', default=0.5, type=float)
    parser.add_argument('--min_lr', default=1e-9, type=float)
    parser.add_argument('--lr', type=float, default=5e-4)

    ## Training
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--log_root', type=str, default='./logs/classify')
    parser.add_argument('--val_noise', type=float, default=0.015)
    parser.add_argument('--max_epochs', type=int, default=800)
    parser.add_argument('--patch_size', type=int, default=1000)

    args = parser.parse_args()

    seed_all(args.seed)
    main(args)
