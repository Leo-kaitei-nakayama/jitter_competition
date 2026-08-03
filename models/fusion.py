"""
Feature Fusion, Gradient Prediction, and Gradient Fusion modules,
adapted from "Adaptive and Iterative Point Cloud Denoising with
Score-Based Diffusion Model" (Wang et al.) for the ASDN codebase.

Built bottom-up:
  1. GradientPrediction  (Eq. 11) -- per-neighbor gradient + importance weight
  2. GradientFusion       (Eq. 12) -- SoftMax-weighted fusion of gradients
  3. FeatureFusion        (Eq. 10) -- fuse E(x^t) and E(x^T) with timestep
"""

import jittor as jt
import jittor.nn as nn
import numpy as np

class ResBlock(nn.Module):
    """standard residual MLP block"""
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        
    def execute(self, x):
        return nn.relu(x + self.mlp(x))
    
class GradientPrediction(nn.Module):
    """
    Eq. 11:  g_{v,i}, w_{v,i} = G(v - x^t_i, F_{t,i})

    For a query point v and one of its neighbors x^t_i (with fused feature
    F_{t,i}), predict a gradient-vector candidate g (3D) and an importance
    weight w (scalar). Implemented as 4 residual blocks over a joint
    embedding of the relative coordinate and the neighbor feature.

    Args at call time:
        rel_coord: (..., 3)   v - x^t_i
        feat:      (..., C)   F_{t,i}
    Returns:
        g: (..., 3)   gradient candidate
        w: (..., 1)   importance weight (raw logit; SoftMax applied later)
    """
    
    def __init__(self, feat_dim, hidden_dim=128, n_blocks=4):
        super().__init__()
        self.input_proj = nn.Linear(3 + feat_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(n_blocks)])
        self.head_g = nn.Linear(hidden_dim, 3)
        self.head_w = nn.Linear(hidden_dim, 1)
        
    def execute(self, rel_coord, feat):
        x = jt.concat([rel_coord, feat], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        g = self.head_g(x)
        w = self.head_w(x)
        return g, w
    
class GradientFusion(nn.Module):
    """
    Eq. 12:  ξ(v) = Σ_i g_{v,i} · SoftMax({w_{v,i}})_i

    Given per-neighbor gradient candidates and their importance weights,
    fuse them into one score vector per query point using a SoftMax over
    the neighbor axis. The SoftMax down-weights uncertain/wrong neighbor
    opinions (e.g. points on the opposite side of a thin structure),
    preventing collapse that a plain average would cause.

    Args at call time:
        g: (B, N, k, 3)   gradient candidates from k neighbors
        w: (B, N, k, 1)   importance weights (raw logits)
    Returns:
        score: (B, N, 3)  fused gradient (score vector) per point
    """
    def execute(self, g, w):
        attn = nn.softmax(w, dim=2)
        score = jt.sum(g * attn, dim = 2)
        return score
    
class FeatureFusion(nn.Module):
    """
    Eq. 10:  F_t = MLP( E(x^t)·e^t + E(x^T)·e^T )

    Blends the feature of the CURRENT iterate E(x^t) with the feature of
    the ORIGINAL noisy cloud E(x^T), modulated by the current position x^t
    and the relative timestep t/T. This lets the network keep the coarse
    shape memory (from x^T) while refining details (from x^t) across
    iterations.

    Args at call time:
        pos:    (B, N, 3)   current point positions x^t
        t_frac: (B, N, 1)   relative timestep t/T, broadcast per point
        feat_t: (B, N, C)   E(x^t), feature of current iterate
        feat_T: (B, N, C)   E(x^T), feature of original noisy cloud
    Returns:
        F: (B, N, C)  fused per-point feature

    Args at __init__:
        gate_mode: how the weight vectors e_t / e_T are produced.
            'pos'     -- e_t = MLP(e), e_T = MLP(e).  The gates depend only on
                         position and timestep, never on the features they gate.
            'posfeat' -- e_t = MLP([e, E(x^t)]), e_T = MLP([e, E(x^T)]).  This is
                         what the paper describes ("the feature e is then fused
                         with E(x^t) to obtain the weight vector e_t, and with
                         E(x^T) to obtain e_T"), and lets the gate react to
                         feature content.
        'pos' is the default only for checkpoint compatibility with models
        trained before 'posfeat' existed; new runs should use 'posfeat'.
    """
    def __init__(self, feat_dim, pos_enc_dim=64, gate_mode='pos'):
        super().__init__()
        assert gate_mode in ('pos', 'posfeat')
        self.gate_mode = gate_mode

        self.pos_mlp = nn.Sequential(
            nn.Linear(3 + 1, pos_enc_dim),
            nn.ReLU(),
            nn.Linear(pos_enc_dim, feat_dim),
        )

        gate_in = feat_dim * 2 if gate_mode == 'posfeat' else feat_dim

        self.weight_t = nn.Sequential(
            nn.Linear(gate_in, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        self.weight_T = nn.Sequential(
            nn.Linear(gate_in, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def execute(self, pos, t_frac, feat_t, feat_T):
        e = self.pos_mlp(jt.concat([pos, t_frac], dim=-1))
        if self.gate_mode == 'posfeat':
            e_t = self.weight_t(jt.concat([e, feat_t], dim=-1))
            e_T = self.weight_T(jt.concat([e, feat_T], dim=-1))
        else:
            e_t = self.weight_t(e)
            e_T = self.weight_T(e)

        fused = feat_t * e_t + feat_T * e_T
        F = self.out_mlp(fused)
        return F
    
    
from .classifyNet import get_knn_idx
from .blocks import gather_neighbors

class FusionHead(nn.Module):
    """
    Full fusion pipeline (paper Figure 2, bottom row):

        E(x^t), E(x^T), x^t, t/T
              │
              ▼
        FeatureFusion  ──►  F_t   (per-point fused feature)
              │
              ▼  (gather k neighbors)
        GradientPrediction  ──►  g_{v,i}, w_{v,i}   (per neighbor)
              │
              ▼
        GradientFusion  ──►  score(v)   (per point)

    Replaces the old linear0_1/2/3 displacement head.

    Args at call time:
        pos:    (B, N, 3)   current positions x^t
        t_frac: (B, N, 1)   relative timestep t/T
        feat_t: (B, N, C)   E(x^t)
        feat_T: (B, N, C)   E(x^T)
    Returns:
        score: (B, N, 3)  predicted displacement / score vector per point
    """
    
    def __init__(self, feat_dim, k=16, grad_hidden=128, n_blocks=4,
                 gate_mode='pos', include_self=False):
        super().__init__()
        self.k = k                        # paper uses k=32
        self.include_self = include_self  # paper samples v from x^t and queries
                                          # neighbours from the same cloud, so the
                                          # point itself is a legitimate x^t_i
        self.feature_fusion = FeatureFusion(feat_dim, gate_mode=gate_mode)
        self.gradient_prediction = GradientPrediction(feat_dim, hidden_dim=grad_hidden, n_blocks=n_blocks)
        self.gradient_fusion = GradientFusion()

    def execute(self, pos, t_frac, feat_t, feat_T):
        B, N, C = feat_t.shape
        F = self.feature_fusion(pos, t_frac, feat_t, feat_T)
        knn_idx = get_knn_idx(pos, pos, self.k, offset=0 if self.include_self else 1)
        F_neighbors = gather_neighbors(F, knn_idx)
        pos_neighbors = gather_neighbors(pos, knn_idx)
        
        rel_coord = pos.unsqueeze(2) - pos_neighbors
        g, w = self.gradient_prediction(rel_coord, F_neighbors)
        
        score = self.gradient_fusion(g, w)
        return score
    
class DiffusionSchedule:
    """
    Linear noise schedule from the paper (Section 4.2):
        T = 1000, beta_0 = 0, beta_T = 2e-6, giving sigma_bar_t in (0, 0.03].

    Provides:
        beta[t], alpha[t], alpha_bar[t], sigma_bar[t]   for t in [0, T]
        step_coef(t, t_delta) -> coefficient in Eq. 7
        find_t_for_sigma(sigma) -> nearest timestep whose sigma_bar matches
        estimate_tau(score_norms) -> adaptive start timestep, Eq. 15 + Eq. 16
    """
    def __init__(self, T=1000, beta_T=2e-6):
        self.T = T
        betas = np.linspace(0.0, beta_T, T+1, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        sigma_bar2 = (1.0 - alpha_bars) / alpha_bars
        sigma_bars = np.sqrt(sigma_bar2)

        self.betas = betas
        self.alpha_bars = alpha_bars
        self.sigma_bar2 = sigma_bar2
        self.sigma_bars = sigma_bars

    def step_coef(self, t, t_delta):
        """Eq. 7 coefficient:  1 - sqrt( (1-ab_{t-d}) ab_t / ((1-ab_t) ab_{t-d}) )"""
        ab_t = self.alpha_bars[t]
        ab_td = self.alpha_bars[t_delta]
        inner = ((1.0 - ab_td)*ab_t) / ((1.0 - ab_t) * ab_td + 1e-20)
        return float(1.0 - np.sqrt(inner))

    def find_t_for_sigma(self, sigma):
        """Nearest timestep t whose sigma_bar[t] is closest to the given sigma."""
        return int(np.argmin(np.abs(self.sigma_bars - sigma)))

    def estimate_tau(self, score_norms, sigma_scale=1.0, estimator='var'):
        """
        Adaptive Schedule Arrangement (paper Section 4.3).

        Eq. 15 estimates the input cloud's noise variance from the network's own
        score prediction on the untouched input:
            sigma_bar^2 = Var( ||s_theta(X|X)|| )
        Eq. 16 then picks the timestep of the TRAINING schedule whose
        sigma_bar^2 is closest to it (searching the discrete {beta_t} rather
        than a continuous beta keeps train and inference schedules aligned --
        paper Appendix E, "w/o align" vs "Full").

        Args:
            score_norms: 1-D numpy array of ||s|| over every point of every patch
            sigma_scale: multiplier on the estimated sigma. Eq. 15's variance-of-
                norms is a biased estimator of the true noise sigma (for isotropic
                noise it underestimates it), so this is the one knob worth
                calibrating on a local eval set built by make_eval_set.py.
            estimator: 'var' -- Eq. 15 as written.
                       'rms' -- sqrt(mean(||s||^2)), which is the unbiased
                                estimate of sigma when the score is the exact
                                displacement to the surface.
        Returns:
            tau: int timestep in [0, T]
        """
        score_norms = np.asarray(score_norms, dtype=np.float64).reshape(-1)
        if estimator == 'rms':
            sigma2 = float((score_norms ** 2).mean())
        else:
            sigma2 = float(score_norms.var())
        sigma2 *= float(sigma_scale) ** 2
        return int(np.argmin(np.abs(self.sigma_bar2 - sigma2)))

