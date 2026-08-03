import jittor as jt
import jittor.nn as nn

from .pointops_jt import knn_points
from .dynamic_edge_conv import DynamicEdgeConv


def get_knn_idx(x, y, k, offset=0):
    """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, M, k) — indices into x's N dimension, for each of y's M query points
    """
    _, knn_idx, _ = knn_points(y, x, K=k + offset)
    return knn_idx[:, :, offset:]


class ScaleNet(nn.Module):
    def __init__(self, k=32, input_dim=0, z_dim=0, embedding_dim=512, output_dim=1):
        super().__init__()
        self.k = k
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.conv1 = DynamicEdgeConv(3, 16)
        self.conv2 = DynamicEdgeConv(16, 24)
        self.conv3 = DynamicEdgeConv(24, 72)
        self.conv4 = DynamicEdgeConv(16 + 24 + 72, self.embedding_dim)

        self.linear1 = nn.Linear(self.embedding_dim, 128, bias=False)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, self.output_dim)

        if self.z_dim > 0:
            self.linear_proj = nn.Linear(256, self.z_dim)
            self.dropout_proj = nn.Dropout(0.1)

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def get_edge_index(self, x):
        """
        Returns (B, N, k) neighbor indices, self excluded.

        NOTE: the original PyTorch version requests k+1 neighbors and calls
        remove_self_loops() on the resulting edge_index to drop the literal
        self-edges. Here we instead request k+1 neighbors (ascending distance,
        so a point's exact match to itself has distance 0 and sorts first)
        and drop index 0 directly, which is equivalent whenever a query point
        is present in its own reference set (always true here, since x is
        queried against itself) and no other point coincides exactly with it.
        """
        idx_full = get_knn_idx(x, x, self.k + 1)  # (B, N, k+1)
        return idx_full[:, :, 1:]  # (B, N, k) — drop self

    def execute(self, x, disp_feat):
        batch_size = x.shape[0]
        num_points = x.shape[1]

        if disp_feat is not None:
            disp_feat = nn.relu(self.linear_proj(disp_feat))
            disp_feat = self.dropout_proj(disp_feat)
            x = jt.concat([x, disp_feat], dim=-1)

        edge_index = self.get_edge_index(x)
        x1 = self.conv1(x, edge_index)  # (B, N, 16)

        edge_index = self.get_edge_index(x1)
        x2 = self.conv2(x1, edge_index)  # (B, N, 24)

        edge_index = self.get_edge_index(x2)
        x3 = self.conv3(x2, edge_index)  # (B, N, 72)

        edge_index = self.get_edge_index(x3)
        x_combined = jt.concat([x1, x2, x3], dim=-1)  # (B, N, 112)
        x = self.conv4(x_combined, edge_index)  # (B, N, embedding_dim)

        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x))
        x = jt.mean(x, dim=1)  # global average pooling over points

        x = jt.tanh(self.linear3(x))

        if self.z_dim > 0:
            return x, x_combined.transpose(0, 2, 1)
        else:
            return x, None
