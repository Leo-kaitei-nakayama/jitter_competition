import numpy as np
import jittor as jt
import jittor.nn as nn

from .classifyNet import ScaleNet
from .pointops_jt import knn_points, farthest_point_sampling
from .utils import get_entropy_B


class Classify(nn.Module):
    """
    Jittor port of the PyTorch-Lightning Classify module.

    Lightning hooks (configure_optimizers, dataloaders, epoch-end logging)
    live in train_classifier.py; the model keeps the loss and the
    patch-based validation routine.
    """

    def __init__(self, args=None, frame_knn=32):
        super().__init__()
        self.args = args
        self.frame_knn = getattr(args, 'frame_knn', frame_knn) if args is not None else frame_knn

        self.feature_nets = nn.ModuleList()
        input_dim = 3
        z_dim = 0
        self.feature_nets.append(
            ScaleNet(k=self.frame_knn, input_dim=input_dim, z_dim=z_dim,
                     embedding_dim=256, output_dim=1)
        )

    # ------------------------------------------------------------------
    @classmethod
    def load_from_checkpoint(cls, ckpt_path):
        """Load weights from the ORIGINAL torch-lightning .ckpt (name-matched)."""
        import torch  # only to deserialize the file

        ckpt = torch.load(ckpt_path, map_location='cpu')
        hparams = ckpt.get('hyper_parameters', {}) or {}
        args = hparams.get('args', None)

        model = cls(args)
        state_dict = ckpt['state_dict']
        jt_state = model.state_dict()

        skipped = []
        for name, tensor in state_dict.items():
            if name in jt_state:
                jt_state[name] = jt.array(tensor.detach().cpu().numpy())
            else:
                skipped.append(name)
        model.load_state_dict(jt_state)

        if skipped:
            print(f"[Classify.load_from_checkpoint] warning: {len(skipped)} "
                  f"unmatched tensors skipped (e.g. {skipped[:5]})")
        model.eval()
        return model

    # ------------------------------------------------------------------
    def get_supervised_loss_nn(self, pcl_noisy, pcl_clean, pcl_seeds, pcl_std):
        """
        Args:
            pcl_noisy:  (B, N, 3)
            pcl_clean:  (B, M, 3)
            pcl_seeds:  (B, 1, 3)
        The target is the entropy ratio between noisy and clean patches;
        the network regresses it (MSE, sum reduction as in the original).
        """
        B, N_noisy, N_clean = pcl_noisy.shape[0], pcl_noisy.shape[1], pcl_clean.shape[1]

        pcl_seeds_1 = pcl_seeds.repeat(1, N_noisy, 1)
        pcl_noisy = pcl_noisy - pcl_seeds_1
        pcl_seeds_2 = pcl_seeds.repeat(1, N_clean, 1)
        pcl_clean = pcl_clean - pcl_seeds_2

        NoiseEntropy = get_entropy_B(pcl_noisy)
        CleanEntropy = get_entropy_B(pcl_clean)

        entropy_ratio = NoiseEntropy / CleanEntropy

        pre, _ = self.feature_nets[0](pcl_noisy, None)
        diff = pre.squeeze(-1) - entropy_ratio
        loss = (diff ** 2).sum()  # MSELoss(reduction='sum')

        return loss

    # ------------------------------------------------------------------
    def patch_based_shang(self, pcl_noisy, pcl_clean, patch_size=1000, seed_k=5,
                           seed_k_alpha=10, num_modules_to_use=None):
        """
        Validation routine: predicted entropy-ratio per patch vs. ground truth
        entropy ratio computed from the clean/noisy patches.

        Args:
            pcl_noisy: (N, 3)
            pcl_clean: (N, 3)
        Returns:
            pre_scale (num_patches, 1), scale_gt (num_patches,)
        """
        assert pcl_noisy.ndim == 2, 'The shape of input point cloud must be (N, 3).'
        N, d = pcl_noisy.shape
        pcl_noisy = pcl_noisy.unsqueeze(0)
        pcl_clean = pcl_clean.unsqueeze(0)
        num_patches = int(seed_k * N / patch_size)
        seed_pnts, indices = farthest_point_sampling(pcl_noisy, num_patches)
        seed_clean_pnts = pcl_clean[0][indices[0]].unsqueeze(0)
        _, _, patches = knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
        _, _, patches_clean = knn_points(seed_clean_pnts, pcl_clean, K=patch_size, return_nn=True)
        patches = patches[0]
        patches_clean = patches_clean[0]
        seed_pnts_1 = seed_pnts.squeeze(0).unsqueeze(1).repeat(1, patch_size, 1)
        patches = patches - seed_pnts_1
        seed_pnts_2 = seed_clean_pnts.squeeze(0).unsqueeze(1).repeat(1, patch_size, 1)
        patches_clean = patches_clean - seed_pnts_2
        pre_scale = []

        i = 0
        patch_step = int(N / (seed_k_alpha * patch_size))
        assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
        while i < num_patches:
            curr_patches = patches[i:i + patch_step]
            try:
                patches_denoised_temp, _ = self.feature_nets[0](curr_patches, None)
            except Exception as e:
                print("=" * 100)
                print(e)
                print("=" * 100)
                print("If this is an Out Of Memory error, Seed_k_alpha might need to be "
                      "increased to decrease patch_step.")
                print("=" * 100)
                return
            pre_scale.append(patches_denoised_temp)
            i += patch_step

        pre_scale = jt.concat(pre_scale, dim=0)
        shang_clean = get_entropy_B(patches_clean)
        shang_noise = get_entropy_B(patches)
        scale_gt = shang_noise / shang_clean

        return pre_scale, scale_gt
