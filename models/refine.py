"""
RefineHead -- a small trainable stage that sits on top of the frozen denoiser.

Why this can work at all: check_learnable.py measured the frozen model's
leftover surface error and found a spatial autocorrelation of +0.35. A point's
error resembles its neighbours' errors, so roughly 12% of the error variance is
predictable from local context alone. Errors with that much structure are
learnable; errors without it are not, which is why the fixed jet-projection
filter failed.

The head sees two things per point:

    the denoised position    where the frozen model put it
    E(x), 32 dims            the frozen encoder's feature AT that position,
                             which carries local shape information the three
                             output coordinates have already discarded

and predicts a corrective displacement. A couple of EdgeConv layers give it the
neighbourhood receptive field the autocorrelation says it needs -- a per-point
MLP would see none of the structure that makes this possible.

The final layer is zero-initialised, so an untrained head predicts exactly zero
and the refined output equals the frozen output bit for bit. Training can only
move away from that starting point, so the stage cannot be worse than not
having it except through training itself. Gradients still flow normally: with
W = 0 the forward output is zero but dL/dW = x·delta is not.
"""

import jittor as jt
import jittor.nn as nn

from .classifyNet import get_knn_idx
from .dynamic_edge_conv import EdgeConv


class RefineHead(nn.Module):
    """
    pos:  (B, N, 3)   denoised positions, centered on the patch seed
    feat: (B, N, C)   E(x) from the frozen encoder at those positions
    ->    (B, N, 3)   corrective displacement, zero at initialisation
    """

    def __init__(self, feat_dim=32, hidden=64, k=16, n_layers=2):
        super().__init__()
        self.k = k
        self.input_proj = nn.Linear(3 + feat_dim, hidden)
        self.convs = nn.ModuleList([EdgeConv(hidden, hidden) for _ in range(n_layers)])
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

        # start as the identity: an untrained head leaves the frozen output alone
        last = self.out[-1]
        last.weight.assign(jt.zeros_like(last.weight))
        if last.bias is not None:
            last.bias.assign(jt.zeros_like(last.bias))

    def execute(self, pos, feat):
        B, N, _ = pos.shape
        x = self.input_proj(jt.concat([pos, feat], dim=-1))
        knn_idx = get_knn_idx(pos, pos, self.k, offset=1)
        for conv in self.convs:
            x = conv(x, knn_idx)
        return self.out(x)
