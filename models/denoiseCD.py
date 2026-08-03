import numpy as np
import jittor as jt
import jittor.nn as nn

from .feature import FeatureExtraction
from .pointops_jt import knn_points, farthest_point_sampling
from .InfoCD import calc_cd_like_InfoV2


class DenoiseNetCD(nn.Module):
    """
    Jittor port of the PyTorch-Lightning DenoiseNetCD.

    Lightning hooks (configure_optimizers, train/val_dataloader,
    training_step, epoch-end logging) are deliberately NOT in the model —
    plain-Jittor equivalents live in train_ASDN.py. The model itself keeps:
        - get_supervised_loss   (training loss)
        - patch_based_denoise   (inference over a big cloud, patch stitching)
        - denoise_langevin_dynamics (per-patch forward pass)
    """

    def __init__(self, args=None, classify_ckpt=None, classify_frame_knn=32,
                 fusion_k=16, fusion_gate='pos', fusion_include_self=False,
                 static_depth=False):
        super().__init__()
        self.args = args
        self.feature_nets = FeatureExtraction(
            classify_ckpt=classify_ckpt, classify_frame_knn=classify_frame_knn,
            fusion_k=fusion_k, fusion_gate=fusion_gate,
            fusion_include_self=fusion_include_self, static_depth=static_depth)
        from .fusion import DiffusionSchedule
        self.schedule = DiffusionSchedule()

    # ------------------------------------------------------------------
    # Checkpoint loading (from the ORIGINAL torch-lightning .ckpt)
    # ------------------------------------------------------------------
    @classmethod
    def load_from_checkpoint(cls, ckpt_path):
        import torch  # only for deserializing the .ckpt file

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
            print(f"[DenoiseNetCD.load_from_checkpoint] warning: {len(skipped)} "
                  f"unmatched tensors skipped (e.g. {skipped[:5]})")
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------
    def curr_iter_add_noise(self, pcl_clean, noise_std):
        new_pcl_clean = pcl_clean + jt.randn_like(pcl_clean) * noise_std.unsqueeze(1).unsqueeze(2)
        return new_pcl_clean.float32()
    
    def compute_gt_score(self, pcl, pcl_clean):
        """
        Eq. 14:  S(x) = NN(x, x_clean) - x
        For each point in pcl, find its nearest neighbor in pcl_clean, and
        return the displacement vector pointing to it (the ground-truth score).

        Args:
            pcl:       (B, N, 3)  points to score (noisy or partially denoised)
            pcl_clean: (B, M, 3)  clean reference points
        Returns:
            score: (B, N, 3)  NN(pcl, pcl_clean) - pcl
        """
        _, _, nn_pts = knn_points(pcl, pcl_clean, K=1, return_nn=True)
        nn_pts = nn_pts.squeeze(2)
        score = nn_pts - pcl
        return score

    def get_supervised_loss(self, pcl_noisy, pcl_clean, pcl_seeds, pcl_std, lam=0.99,
                             mask_size=256, t_min=30, t_norm='T'):
        """
        Two-stage sampling loss (paper Algorithm 1, Eq. 9).
            Stage 1: predict score at x^t, loss vs GT score S(x^t)
            Stage 2: step to x^{t-Δ} using Stage-1 score (Eq. 7),
                     predict again with feat_T = E(x^t) cached from Stage 1

        Args:
            mask_size: K_p in Algorithm 1 -- how many of the patch's innermost
                points carry the loss. The bridge returns each patch sorted by
                distance from its seed, so the mask is simply the first
                `mask_size` rows. Pass 0 to disable masking.
            t_min: floor on the sampled timesteps. The Eq. 9 weight
                (1-λ)/σ̄_t + λ diverges as t → 0, so both t and t-Δ are kept
                above this. Corresponds to the paper's t ~ U({t_min,...,T}).
                Keep it at or below τ̂_min/L so training covers every timestep
                inference visits: the quietest competition clouds (σ=0.005) give
                τ̂≈158, and an L=5 trajectory evaluates the score down to t≈32.
            t_norm: what the relative timestep fed to the feature fusion module
                is divided by.
                'T'   -- t/T, i.e. absolute position in the training schedule.
                'tau' -- t/t, the paper's convention (Algorithm 1 lines 7/11 use
                         t/t and (t-Δ)/t; Algorithm 2 line 10 uses t/τ̂), which
                         makes the value mean "progress through this trajectory"
                         and always starts at 1.
                This must match whatever predict_on_starter.py is given.
        """
        B, N_noisy, N_clean = pcl_noisy.shape[0], pcl_noisy.shape[1], pcl_clean.shape[1]

        # center on seeds (same as before)
        pcl_noisy = pcl_noisy - pcl_seeds.repeat(1, N_noisy, 1)
        pcl_clean = pcl_clean - pcl_seeds.repeat(1, N_clean, 1)

        offset = jt.array(np.array([(i + 1) * N_noisy for i in range(B)]), dtype='int32')
        feat_empty = pcl_noisy.reshape(B * N_noisy, -1)[:, 3:]

        # --- map each sample's noise_std to its OWN diffusion timestep t ---
        # (the old code collapsed the batch to one t via pcl_std.mean(), so a
        # batch mixing 0.005 and 0.02 noise trained every sample at the mean)
        std_np = pcl_std.numpy().reshape(-1)
        t_np = np.empty((B,), dtype=np.int64)
        t_tgt_np = np.empty((B,), dtype=np.int64)
        for i in range(B):
            t_i = max(self.schedule.find_t_for_sigma(float(std_np[i])), t_min + 1)
            # Δ ~ Uniform({1,...,t}) per Algorithm 1 line 9, floored at t_min.
            # The old code always used Δ = t/2, so the network only ever saw one
            # step size while inference walks a τ̂/L grid.
            t_np[i] = t_i
            t_tgt_np[i] = np.random.randint(t_min, t_i)

        sigma_t = self.schedule.sigma_bars[t_np]        # (B,)
        sigma_td = self.schedule.sigma_bars[t_tgt_np]   # (B,)
        coef_np = np.array([self.schedule.step_coef(int(t_np[i]), int(t_tgt_np[i]))
                            for i in range(B)], dtype=np.float32)

        if t_norm == 'tau':
            frac1_np = np.ones((B,), dtype=np.float32)
            frac2_np = (t_tgt_np / np.maximum(t_np, 1)).astype(np.float32)
        else:
            frac1_np = (t_np / self.schedule.T).astype(np.float32)
            frac2_np = (t_tgt_np / self.schedule.T).astype(np.float32)

        def _per_sample(arr):
            """(B,) numpy -> (B, N, 1) jt.Var, broadcastable over coords."""
            v = jt.array(np.asarray(arr, dtype=np.float32).reshape(B, 1, 1))
            return v.broadcast((B, N_noisy, 1))

        t_frac = _per_sample(frac1_np)
        t_frac_td = _per_sample(frac2_np)
        coef = _per_sample(coef_np)
        w1 = _per_sample((1.0 - lam) / sigma_t + lam)
        w2 = _per_sample((1.0 - lam) / sigma_td + lam)

        # --- patch mask M (Algorithm 1 line 4) ---
        # Only the K_p innermost points are scored, and in Stage 2 the outer ring
        # is advanced with the GROUND-TRUTH score instead of the predicted one.
        # That keeps every scored point surrounded by a well-formed neighbourhood,
        # which is exactly what the gradient-prediction/fusion pair needs.
        # Without it, patch-boundary points -- whose KNN support is truncated and
        # whose nearest clean neighbour may lie outside pcl_clean -- carried the
        # same weight as the well-supported centre.
        use_mask = 0 < mask_size < N_noisy
        if use_mask:
            mask_np = np.zeros((B, N_noisy), dtype=np.float32)
            mask_np[:, :mask_size] = 1.0
            mask2 = jt.array(mask_np)                       # (B, N)
            mask3 = jt.array(mask_np.reshape(B, N_noisy, 1))  # (B, N, 1)
            denom = float(B * mask_size)
        else:
            mask2 = mask3 = None
            denom = float(B * N_noisy)

        def _masked_loss(w, pred, gt):
            err = ((w * (pred - gt)) ** 2).sum(dim=-1)   # (B, N)
            if mask2 is not None:
                err = err * mask2
            return err.sum() / denom

        # ================= Stage 1 =================
        x_t = pcl_noisy
        score1, feat_T = self.feature_nets(
            x_t, feat_empty, offset, feat_T=None, t_frac=t_frac, return_feat=True)
        gt_score1 = self.compute_gt_score(x_t, pcl_clean)
        loss1 = _masked_loss(w1, score1, gt_score1)

        # ================= Stage 2 =================
        # step to x^{t-Δ} using the Eq. 7 coefficient. Algorithm 1 line 10 moves
        # masked points by the predicted score and unmasked points by the GT one.
        step_score = score1.detach()
        if mask3 is not None:
            step_score = mask3 * step_score + (1.0 - mask3) * gt_score1
        x_td = (x_t + coef * step_score).detach()  # detach: stage-2 input is a fixed cloud

        score2 = self.feature_nets(
            x_td, feat_empty, offset, feat_T=feat_T, t_frac=t_frac_td, return_feat=False)
        gt_score2 = self.compute_gt_score(x_td, pcl_clean)
        loss2 = _masked_loss(w2, score2, gt_score2)

        return loss1 + loss2

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def patch_based_denoise(self, pcl_noisy, patch_size=1000, seed_k=5,
                             seed_k_alpha=10, num_modules_to_use=None):
        """
        Args:
            pcl_noisy: (N, 3)
        """
        assert pcl_noisy.ndim == 2, 'The shape of input point cloud must be (N, 3).'
        N, d = pcl_noisy.shape
        pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
        num_patches = int(seed_k * N / patch_size)
        seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
        patch_dists, point_idxs_in_main_pcd, patches = knn_points(
            seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
        patches = patches[0]  # (num_patches, K, 3)

        # Patch stitching preliminaries
        seed_pnts_1 = seed_pnts.squeeze(0).unsqueeze(1).repeat(1, patch_size, 1)
        patches = patches - seed_pnts_1
        patch_dists, point_idxs_in_main_pcd = patch_dists[0], point_idxs_in_main_pcd[0]
        patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(1, patch_size)

        # For each original point, distance-derived weight per covering patch
        all_dists_np = np.full((num_patches, N), np.inf, dtype=np.float32)
        pid_np = point_idxs_in_main_pcd.numpy()
        pdist_np = patch_dists.numpy()
        for pi in range(num_patches):
            all_dists_np[pi, pid_np[pi]] = pdist_np[pi]

        weights = np.exp(-1 * all_dists_np)              # (num_patches, N)
        best_weights_idx = weights.argmax(axis=0)         # (N,)

        patches_denoised = []

        # Denoising
        i = 0
        patch_step = int(N / (seed_k_alpha * patch_size))
        assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
        while i < num_patches:
            curr_patches = patches[i:i + patch_step]
            patches_denoised_temp = self.denoise_langevin_dynamics(curr_patches)
            patches_denoised.append(patches_denoised_temp)
            i += patch_step

        patches_denoised = jt.concat(patches_denoised, dim=0)
        patches_denoised = patches_denoised + seed_pnts_1

        # Patch stitching: for each original point, take its denoised coordinate
        # from the patch that covers it with the highest weight.
        # (Original used a per-point boolean-mask list comprehension; Jittor
        # can't index with numpy bool masks, so build integer gather indices:
        # pos_map[pi, n] = position j of original point n inside patch pi.)
        pos_map = np.full((num_patches, N), -1, dtype=np.int64)
        col = np.arange(patch_size, dtype=np.int64)
        for pi in range(num_patches):
            pos_map[pi, pid_np[pi]] = col

        point_ids = np.arange(N, dtype=np.int64)
        gather_pos = pos_map[best_weights_idx, point_ids]      # (N,) position in its best patch
        covered = gather_pos >= 0                               # uncovered points -> pad later

        sel_patch = jt.array(best_weights_idx[covered].astype(np.int32))
        sel_pos = jt.array(gather_pos[covered].astype(np.int32))
        pcl_denoised = patches_denoised[sel_patch, sel_pos]     # (N_covered, 3)

        while pcl_denoised.shape[0] != N:
            pcl_denoised = jt.concat(
                (pcl_denoised, pcl_denoised[pcl_denoised.shape[0] - 1].unsqueeze(0)), dim=0)
            print(f'pcl_denoised.shape ===> {pcl_denoised.shape}')

        return pcl_denoised
    
    def patch_based_denoise_diffusion(self, pcl_noisy, patch_size=1000, seed_k=5,
                                       seed_k_alpha=10, L=5, t_start=632,
                                       adaptive=True, sigma_scale=1.0,
                                       sigma_estimator='var', t_norm='T',
                                       return_tau=False):
        """
        Adaptive and Iterative Denoising (paper Algorithm 2).

        Same patch splitting / stitching as patch_based_denoise, but each patch
        is denoised with L-step diffusion sampling instead of a single forward
        pass, and -- when `adaptive` is set -- the starting timestep τ̂ is
        estimated from the input cloud itself rather than fixed.

        Why adaptive matters here: the schedule is calibrated so that σ̄_t ≈
        3.16e-5·t², i.e. t=632 corresponds to σ=0.02 and t=158 to σ=0.005 --
        exactly the two ends of this competition's noise range. A fixed
        t_start=632 therefore runs the *maximum-noise* schedule on every cloud,
        over-denoising the quiet ones into over-smoothed surfaces. Since the
        score is scored against the noisy input as the zero baseline, that shows
        up directly as lost CD and P2S points.

        Args:
            adaptive: estimate τ̂ per cloud (Eq. 15 + Eq. 16) instead of using
                t_start. Costs one extra forward pass over the cloud.
            sigma_scale, sigma_estimator: see DiffusionSchedule.estimate_tau.
                sigma_scale is the knob to calibrate against a local eval set.
            t_norm: must match what the model was trained with -- see
                get_supervised_loss.
        """
        assert pcl_noisy.ndim == 2, 'The shape of input point cloud must be (N, 3).'
        N, d = pcl_noisy.shape
        pcl_noisy = pcl_noisy.unsqueeze(0)
        num_patches = int(seed_k * N / patch_size)
        seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
        patch_dists, point_idxs_in_main_pcd, patches = knn_points(
            seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
        patches = patches[0]
        seed_pnts_1 = seed_pnts.squeeze(0).unsqueeze(1).repeat(1, patch_size, 1)
        patches = patches - seed_pnts_1
        patch_dists, point_idxs_in_main_pcd = patch_dists[0], point_idxs_in_main_pcd[0]
        patch_dists = patch_dists / patch_dists[:, -1].unsqueeze(1).repeat(1, patch_size)
        all_dists_np = np.full((num_patches, N), np.inf, dtype=np.float32)
        pid_np = point_idxs_in_main_pcd.numpy()
        pdist_np = patch_dists.numpy()
        for pi in range(num_patches):
            all_dists_np[pi, pid_np[pi]] = pdist_np[pi]
        weights = np.exp(-1 * all_dists_np)
        best_weights_idx = weights.argmax(axis=0)

        patch_step = int(N / (seed_k_alpha * patch_size))
        assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
        chunk_starts = list(range(0, num_patches, patch_step))

        # ---- Adaptive Schedule Arrangement (Algorithm 2, lines 2-3) ----
        # ŝ_θ(X|X) is the score the network predicts on the untouched input, i.e.
        # feat_T = E(X) and relative timestep 1. Only the norms are kept, so this
        # probe costs one forward pass and no lasting memory.
        tau = int(t_start)
        if adaptive:
            probe_frac = 1.0 if t_norm == 'tau' else float(t_start) / self.schedule.T
            norms = []
            with jt.no_grad():
                for s in chunk_starts:
                    score, _ = self._patch_forward(
                        patches[s:s + patch_step], feat_T=None, t_frac_val=probe_frac)
                    norms.append(jt.sqrt((score ** 2).sum(dim=-1) + 1e-12).numpy().reshape(-1))
            tau = self.schedule.estimate_tau(
                np.concatenate(norms), sigma_scale=sigma_scale, estimator=sigma_estimator)
        tau = int(min(max(tau, L), self.schedule.T))

        patches_denoised = []
        for s in chunk_starts:
            patches_denoised.append(self.denoise_langevin_dynamics_diffusion(
                patches[s:s + patch_step], L=L, t_start=tau, t_norm=t_norm))
        patches_denoised = jt.concat(patches_denoised, dim=0)
        patches_denoised = patches_denoised + seed_pnts_1
        pos_map = np.full((num_patches, N), -1, dtype=np.int64)
        col = np.arange(patch_size, dtype=np.int64)
        for pi in range(num_patches):
            pos_map[pi, pid_np[pi]] = col
        point_ids = np.arange(N, dtype=np.int64)
        gather_pos = pos_map[best_weights_idx, point_ids]
        covered = gather_pos >= 0
        sel_patch = jt.array(best_weights_idx[covered].astype(np.int32))
        sel_pos = jt.array(gather_pos[covered].astype(np.int32))
        pcl_denoised = patches_denoised[sel_patch, sel_pos]
        while pcl_denoised.shape[0] != N:
            pcl_denoised = jt.concat(
                (pcl_denoised, pcl_denoised[pcl_denoised.shape[0] - 1].unsqueeze(0)), dim=0)
        if return_tau:
            return pcl_denoised, tau
        return pcl_denoised

    def patch_based_denoise_without_stitching(self, pcl_noisy, patch_size=1000,
                                               seed_k=5, seed_k_alpha=10,
                                               num_modules_to_use=None):
        """
        Simpler variant used by test_ASDN.py when --patch_stitching is off:
        denoise patches and concatenate all their points, then FPS back to N.
        """
        assert pcl_noisy.ndim == 2
        N, d = pcl_noisy.shape
        pcl_noisy = pcl_noisy.unsqueeze(0)
        num_patches = int(seed_k * N / patch_size)
        seed_pnts, _ = farthest_point_sampling(pcl_noisy, num_patches)
        _, _, patches = knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
        patches = patches[0]
        seed_pnts_1 = seed_pnts.squeeze(0).unsqueeze(1).repeat(1, patch_size, 1)
        patches = patches - seed_pnts_1

        patches_denoised = []
        i = 0
        patch_step = int(N / (seed_k_alpha * patch_size))
        assert patch_step > 0, "Seed_k_alpha needs to be decreased to increase patch_step!"
        while i < num_patches:
            curr_patches = patches[i:i + patch_step]
            patches_denoised.append(self.denoise_langevin_dynamics(curr_patches))
            i += patch_step

        patches_denoised = jt.concat(patches_denoised, dim=0) + seed_pnts_1
        all_pts = patches_denoised.reshape(1, -1, 3)
        sampled, _ = farthest_point_sampling(all_pts, N)
        return sampled[0]

    def denoise_langevin_dynamics(self, pcl_noisy):
        """
        Args:
            pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        pred_disps = []

        with jt.no_grad():
            self.feature_nets.eval()

            feat = pcl_noisy.reshape(B * N, -1)[:, 3:]
            offset = jt.array(np.array([(i + 1) * N for i in range(B)]), dtype='int32')

            pred_points = self.feature_nets(pcl_noisy, feat, offset)
            pred_disps.append(pred_points)

        return pcl_noisy + pred_disps[-1]

    def _patch_forward(self, patches, feat_T=None, t_frac_val=None):
        """
        One network pass over a (B, K, 3) batch of centered patches.

        Returns (score, feat) where feat is E(patches). Note that E does not
        depend on t_frac -- the relative timestep only enters the fusion head --
        so a call with feat_T=None yields both E(x) and ŝ(x|x) at once.
        """
        B, K, _ = patches.shape
        offset = jt.array(np.array([(i + 1) * K for i in range(B)]), dtype='int32')
        feat_in = patches.reshape(B * K, -1)[:, 3:]
        t_frac = None
        if t_frac_val is not None:
            t_frac = jt.full((B, K, 1), float(t_frac_val)).float32()
        return self.feature_nets(patches, feat_in, offset,
                                 feat_T=feat_T, t_frac=t_frac, return_feat=True)

    def denoise_langevin_dynamics_diffusion(self, patches, L=5, t_start=632, t_norm='T'):
        """
        Diffusion-style iterative denoising for a batch of patches (paper Alg. 2,
        lines 5-13). Caches E(x^τ̂) once (original patch feature) and runs L
        reverse steps, moving points via Eq. 7's step_coef at each step.

        Args:
            patches: (B, K, 3) centered patches (same as denoise_langevin_dynamics input)
            t_start: τ̂, the starting timestep. patch_based_denoise_diffusion
                estimates this per cloud; passing it directly gives the paper's
                non-adaptive "FixedSched" baseline.
        Returns:
            (B, K, 3) denoised patches
        """
        tau = int(max(t_start, L))
        # Algorithm 2 line 8: t = Round(l·Δ) for l = L..0, with Δ = τ̂/L.
        step_ts = [int(round(l * tau / L)) for l in range(L, -1, -1)]

        x_t = patches
        feat_T = None
        with jt.no_grad():
            for i in range(L):
                t, t_next = step_ts[i], step_ts[i + 1]
                t_frac_val = (t / tau) if t_norm == 'tau' else (t / self.schedule.T)

                if i == 0:
                    # x^t is still the original patch here, so this single pass
                    # yields E(x^τ̂) -- cached for every later step -- alongside
                    # the first score. No separate feature-extraction pass needed.
                    score, feat_T = self._patch_forward(
                        x_t, feat_T=None, t_frac_val=t_frac_val)
                else:
                    score, _ = self._patch_forward(
                        x_t, feat_T=feat_T, t_frac_val=t_frac_val)

                x_t = x_t + self.schedule.step_coef(t, t_next) * score

        return x_t