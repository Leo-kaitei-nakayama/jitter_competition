import jittor as jt
import jittor.nn as nn


class EdgeConv(nn.Module):
    """
    out_i = MLP_lin(x_i) + max_{j in N(i)} MLP_mlp([x_i, x_j - x_i])

    Same formula as the PyG-based original (aggr='max'), but computed on a
    dense (B, N, k, C) neighbor tensor instead of a sparse edge_index, since
    Jittor has no torch_geometric / torch_scatter dependency available.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.lin = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def execute(self, x, knn_idx):
        """
        Args:
            x:       (B, N, C_in)
            knn_idx: (B, N, k) neighbor indices into the N dimension of x
                     (self excluded — see classifyNet.get_edge_index)
        Returns:
            (B, N, C_out)
        """
        B, N, C_in = x.shape
        k = knn_idx.shape[-1]

        # gather neighbor features: x_j -> (B, N, k, C_in)
        batch_idx = jt.arange(B).reshape(B, 1, 1).broadcast((B, N, k))
        x_j = x[batch_idx, knn_idx]  # (B, N, k, C_in)

        x_i = x.unsqueeze(2).broadcast((B, N, k, C_in))  # (B, N, k, C_in)

        tmp = jt.concat([x_i, x_j - x_i], dim=-1)  # (B, N, k, 2*C_in)
        tmp = tmp.reshape(B * N * k, 2 * C_in)
        msg = self.mlp(tmp).reshape(B, N, k, self.out_channels)

        out_1 = jt.max(msg, dim=2)  # max aggregation over neighbors -> (B, N, C_out)

        out_2 = self.lin(x.reshape(B * N, C_in)).reshape(B, N, self.out_channels)

        return out_1 + out_2


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def execute(self, x, knn_idx):
        return super().execute(x, knn_idx)
