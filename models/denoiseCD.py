import numpy as np
import jittor as jt
import jittor.nn as nn

from .feature import FeatureExtraction
from .pointops_jt import knn_points, farthest_point_sampling
from .classifyNet import get_knn_idx
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
                 static_depth=False, max_sigma=None):
        super().__init__()
        self.args = args
        self.feature_nets = FeatureExtraction(
            classify_ckpt=classify_ckpt, classify_frame_knn=classify_frame_knn,
            fusion_k=fusion_k, fusion_gate=fusion_gate,
            fusion_include_self=fusion_include_self, static_depth=static_depth)
        from .fusion import DiffusionSchedule
        # max_sigma sets the largest noise level the schedule can represent.
        # Must match between training and inference -- see DiffusionSchedule.
        self.schedule = DiffusionSchedule(max_sigma=max_sigma)

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

    @staticmethod
    def spacing_uniformity(pts):
        """Relative variance of nearest-neighbour spacing over pts (B, M, 3).

        Scale-free (it divides by the mean spacing), so it does not fight the
        squared-distance score term for control of the overall scale, and it is
        zero exactly when every point is equidistant from its nearest
        neighbour. Neighbour indices are treated as constant; the gradient
        reaches the caller through the positions.
        """
        B, M, _ = pts.shape
        idx = get_knn_idx(pts, pts, k=1, offset=1).reshape(B, M)
        bidx = jt.arange(B).reshape(B, 1).repeat(1, M)
        d1 = jt.sqrt(((pts[bidx, idx] - pts) ** 2).sum(dim=-1) + 1e-12)
        dm = d1.mean(dim=1, keepdims=True)
        return (((d1 - dm) / (dm + 1e-12)) ** 2).mean()

    def get_supervised_loss(self, pcl_noisy, pcl_clean, pcl_seeds, pcl_std, lam=0.99,
                             mask_size=256, t_min=30, t_norm='T', exact_score=False,
                             unif_weight=0.0, cov_weight=0.0):
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
            exact_score: use the exact per-point displacement as the training
                target instead of Eq. 14's nearest-neighbour approximation.
                Eq. 14 defines S(x) = NN(x, x_clean) - x because in the paper's
                setting the clean cloud is an independent sampling of the surface
                with no correspondence to the noisy one. This competition's data
                is different: bridge/data_bridge.py builds both patches from the
                SAME point indices, so row i of pcl_clean is row i of pcl_noisy
                before displacement, and the true score is simply
                pcl_clean - pcl. The nearest-neighbour search approximates a
                quantity that is already known exactly, and it returns the wrong
                point whenever a displaced sample lands closer to a neighbour
                than to its own origin -- which pulls points sideways and
                clusters them.
                Training-only; inference is unaffected, so predict_on_starter.py
                needs no matching flag.
            unif_weight: weight of a point-spacing uniformity penalty on x + ŝ,
                i.e. on where the network wants each point to end up.

                The score target NN(x, clean) - x is not injective: several
                noisy points can share a nearest clean point, and the loss above
                is fully satisfied when they all land on it. That degeneracy is
                invisible to a per-point squared error but is exactly what the
                CD metric's second term punishes, which is why CD trails P2S by
                ~28 points. This term is the missing signal, applied where the
                scramble originates rather than in a downstream stage that can
                only tidy up afterwards.

                Scale note: the score term is a squared distance (~1e-3 for this
                data) while this one is a relative variance (~1e-1), so start
                around 1e-5..1e-3 and sweep. 0 reproduces the original loss
                exactly.
            cov_weight: weight of CD's SECOND term, applied at x + ŝ. The score
                loss above already IS CD's first term (pull each sent point to
                its nearest clean point); what it misses is the reverse demand
                that every clean point have a sent point nearby -- the coverage
                signal that punishes collapse directly, where the uniformity
                term is only a proxy for it. Measured over the central
                mask_size clean rows (rows are seed-distance sorted), against
                ALL sent points, so patch-edge truncation cannot fake holes.
                Scale note: this term has the same units as a squared distance
                and floors near (half point spacing)^2 ~ 6e-6 even for perfect
                coverage, so it needs a LARGE weight to matter: sweep 3..30.
        """
        B, N_noisy, N_clean = pcl_noisy.shape[0], pcl_noisy.shape[1], pcl_clean.shape[1]
        if exact_score and N_noisy != N_clean:
            raise ValueError(
                f'exact_score needs point-corresponded patches, but got '
                f'{N_noisy} noisy vs {N_clean} clean points. Only use it with a '
                f'dataset that pairs the two row-by-row.')

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

        def _gt_score(pcl):
            # exact: row-corresponded displacement. approximate: Eq. 14's NN search.
            return (pcl_clean - pcl) if exact_score else self.compute_gt_score(pcl, pcl_clean)

        # ================= Stage 1 =================
        x_t = pcl_noisy
        score1, feat_T = self.feature_nets(
            x_t, feat_empty, offset, feat_T=None, t_frac=t_frac, return_feat=True)
        gt_score1 = _gt_score(x_t)
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
        gt_score2 = _gt_score(x_td)
        loss2 = _masked_loss(w2, score2, gt_score2)

        loss = loss1 + loss2
        # exposed for logging: without the split, a weak uniformity term is
        # indistinguishable from no term at all in the total
        self.last_score_loss = float(loss.item())
        self.last_unif_raw = 0.0
        if unif_weight > 0:
            # x + ŝ is where the network is sending each point. Penalising the
            # spread of nearest-neighbour spacing there is a direct penalty on
            # several points being sent to the same place -- the degeneracy the
            # per-point score loss cannot see. Measured on the central mask
            # only: patch-edge points have truncated neighbourhoods and their
            # spacing carries no usable signal.
            M = mask_size if use_mask else N_noisy
            unif = (self.spacing_uniformity((x_t + score1)[:, :M, :])
                    + self.spacing_uniformity((x_td + score2)[:, :M, :]))
            self.last_unif_raw = float(unif.item())
            loss = loss + unif_weight * unif

        self.last_cov_raw = 0.0
        if cov_weight > 0:
            M = mask_size if use_mask else N_noisy
            clean_c = pcl_clean[:, :M, :]
            bidx = jt.arange(B).reshape(B, 1).repeat(1, M)

            def _coverage(sent):
                # nearest SENT point for each central clean point; indices are
                # constants, the distance is recomputed so the gradient pulls
                # that sent point toward the uncovered clean point
                _, idx, _ = knn_points(clean_c, sent, K=1)
                nearest = sent[bidx, idx[:, :, 0]]
                return ((clean_c - nearest) ** 2).sum(dim=-1).mean()

            cov = _coverage(x_t + score1) + _coverage(x_td + score2)
            self.last_cov_raw = float(cov.item())
            loss = loss + cov_weight * cov
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @staticmethod
    def select_stitch_source(pid_np, pdist_np):
        """Pick, for every cloud point, which patch's denoised coordinate to keep.

        Replaces the dense construction all three stitching arrays shared:

            all_dists = np.full((num_patches, N), inf); all_dists[pi, pid[pi]] = ...
            best = np.exp(-all_dists).argmax(axis=0)
            pos_map = np.full((num_patches, N), -1); ...

        Those are O(num_patches * N) = O(0.006 * N^2) bytes -- 240 MB at N=50k,
        24 GB at N=500k -- while >99% of the entries are the inf/-1 filler,
        because each patch covers only patch_size of the N points. This walks
        the (num_patches, patch_size) covering pairs instead: O(seed_k * N).

        Equivalence with the dense argmax, including its tie-breaks: lexsort is
        stable and the flat array is patch-major, so among equal (point, weight)
        pairs the lowest patch index wins -- exactly what argmax returned. The
        weight is computed with the same float32 exp. Uncovered points simply
        never appear, matching the pos_map[argmax]= -1 exclusion; the caller's
        padding loop handles them as before.

        Args:
            pid_np:   (num_patches, patch_size) int   -- point ids per patch
            pdist_np: (num_patches, patch_size) float -- normalized distances
        Returns:
            sel_patch (C,) int32, sel_pos (C,) int32: source patch and row for
            each covered point, ordered by ascending point id.
        """
        K = pid_np.shape[1]
        flat_pid = pid_np.reshape(-1)
        flat_w = np.exp(-pdist_np.reshape(-1).astype(np.float32))
        order = np.lexsort((-flat_w, flat_pid))
        pid_sorted = flat_pid[order]
        keep = np.ones(order.size, dtype=bool)
        keep[1:] = pid_sorted[1:] != pid_sorted[:-1]
        sel = order[keep]
        return ((sel // K).astype(np.int32), (sel % K).astype(np.int32))

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

        # For each original point, the best covering patch (linear memory)
        pid_np = point_idxs_in_main_pcd.numpy()
        pdist_np = patch_dists.numpy()
        sel_patch_np, sel_pos_np = self.select_stitch_source(pid_np, pdist_np)

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
        pcl_denoised = patches_denoised[jt.array(sel_patch_np), jt.array(sel_pos_np)]

        while pcl_denoised.shape[0] != N:
            pcl_denoised = jt.concat(
                (pcl_denoised, pcl_denoised[pcl_denoised.shape[0] - 1].unsqueeze(0)), dim=0)
            print(f'pcl_denoised.shape ===> {pcl_denoised.shape}')

        return pcl_denoised
    
    def patch_based_denoise_diffusion(self, pcl_noisy, patch_size=1000, seed_k=5,
                                       seed_k_alpha=10, L=5, t_start=632,
                                       adaptive=True, sigma_scale=1.0,
                                       sigma_estimator='var', t_norm='T',
                                       return_tau=False, refine_head=None,
                                       stop_frac=0.0, straight=False,
                                       stitch='best', stitch_alpha=1.0):
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
        pid_np = point_idxs_in_main_pcd.numpy()
        pdist_np = patch_dists.numpy()
        sel_patch_np, sel_pos_np = self.select_stitch_source(pid_np, pdist_np)

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
                patches[s:s + patch_step], L=L, t_start=tau, t_norm=t_norm,
                refine_head=refine_head, stop_frac=stop_frac, straight=straight))
        patches_denoised = jt.concat(patches_denoised, dim=0)
        patches_denoised = patches_denoised + seed_pnts_1
        if stitch == 'mean':
            # Weighted average over every patch that covers a point, instead of
            # winner-take-all. Each patch's estimate of a point carries
            # independent patch-placement noise, so the ~seed_k-fold overlap is
            # a free ensemble; stitch_alpha sharpens the weights so unreliable
            # patch-edge predictions count less (alpha=1 mirrors the exp(-d)
            # weights the argmax used; larger alpha approaches 'best').
            pd = patches_denoised.numpy().astype(np.float64)      # (P, K, 3)
            w = np.exp(-stitch_alpha * pdist_np.astype(np.float64))
            acc = np.zeros((N, 3), np.float64)
            wsum = np.zeros((N,), np.float64)
            np.add.at(acc, pid_np.reshape(-1),
                      pd.reshape(-1, 3) * w.reshape(-1, 1))
            np.add.at(wsum, pid_np.reshape(-1), w.reshape(-1))
            covered = wsum > 0
            pcl_denoised = jt.array(
                (acc[covered] / wsum[covered, None]).astype(np.float32))
        else:
            pcl_denoised = patches_denoised[jt.array(sel_patch_np), jt.array(sel_pos_np)]
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

    def refine_patches(self, patches, refine_head):
        """
        Apply a trained RefineHead to already-denoised patches.

        Runs the frozen encoder once more so the features line up with the
        denoised positions rather than the input ones, then adds the head's
        corrective displacement. Kept separate from the sampler so the same code
        path serves training and inference.

        patches: (B, K, 3) denoised, still centered on their seeds.
        """
        if refine_head is None:
            return patches
        with jt.no_grad():
            _, feat = self._patch_forward(patches, feat_T=None, t_frac_val=0.0)
        return patches + refine_head(patches, feat)

    def denoise_langevin_dynamics_diffusion(self, patches, L=5, t_start=632, t_norm='T',
                                             refine_head=None, return_traj=False,
                                             stop_frac=0.0, straight=False):
        """
        Diffusion-style iterative denoising for a batch of patches (paper Alg. 2,
        lines 5-13). Caches E(x^τ̂) once (original patch feature) and runs L
        reverse steps, moving points via Eq. 7's step_coef at each step.

        Args:
            patches: (B, K, 3) centered patches (same as denoise_langevin_dynamics input)
            t_start: τ̂, the starting timestep. patch_based_denoise_diffusion
                estimates this per cloud; passing it directly gives the paper's
                non-adaptive "FixedSched" baseline.
            return_traj: also return, per step, the positions after the step and
                the per-point score norms that produced it -- for
                measure_iteration.py. The denoising math is untouched.
            stop_frac: per-point early stopping (ASDN's stop-in-advance idea at
                point granularity, driven by the score instead of the entropy
                classifier). From step 2 on, a point whose predicted score norm
                has fallen below stop_frac x (its patch's mean step-1 score
                norm) is frozen for the remaining steps -- it is already where
                the model wants it, and further steps only add tangential
                drift. 0 disables. Sweep against the tune split after
                measure_iteration.py confirms the overshoot exists.
            straight: StraightPCF-inspired: restrict steps 2..L to the
                direction each point moved in step 1, killing the zigzag
                component of the trajectory. Off by default.
        Returns:
            (B, K, 3) denoised patches
            [if return_traj] (denoised, traj) where traj is a list of L tuples
                (positions_after_step (B,K,3) np, score_norm (B,K) np)
        """
        tau = int(max(t_start, L))
        # Algorithm 2 line 8: t = Round(l·Δ) for l = L..0, with Δ = τ̂/L.
        step_ts = [int(round(l * tau / L)) for l in range(L, -1, -1)]

        x_t = patches
        feat_T = None
        traj = []
        d0_unit = None
        ref_norm = None
        frozen = None
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

                step = self.schedule.step_coef(t, t_next) * score

                if (stop_frac > 0 or straight) and i == 0:
                    norm0 = jt.sqrt((score ** 2).sum(dim=-1) + 1e-12)   # (B, K)
                    if stop_frac > 0:
                        ref_norm = norm0.mean(dim=1, keepdims=True)      # (B, 1)
                        frozen = jt.zeros_like(norm0)
                    if straight:
                        d0_unit = step / jt.sqrt(
                            (step ** 2).sum(dim=-1, keepdims=True) + 1e-24)

                if i > 0:
                    if straight:
                        along = (step * d0_unit).sum(dim=-1, keepdims=True)
                        step = along * d0_unit
                    if stop_frac > 0:
                        norm_i = jt.sqrt((score ** 2).sum(dim=-1) + 1e-12)
                        frozen = jt.maximum(
                            frozen, (norm_i < stop_frac * ref_norm).float32())
                        step = step * (1.0 - frozen).unsqueeze(-1)

                x_t = x_t + step
                if return_traj:
                    norm = jt.sqrt((score ** 2).sum(dim=-1) + 1e-12)
                    traj.append((x_t.numpy().copy(), norm.numpy().copy()))

        out = self.refine_patches(x_t, refine_head)
        if return_traj:
            return out, traj
        return out