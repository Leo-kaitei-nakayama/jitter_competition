import jittor as jt
import jittor.nn as nn

from . import pointops_jt as pointops


def block_decider(name):
    if name == 'startblock':
        return StartBlock
    if name == 'upsample':
        return Upsampling
    if name == 'downsample':
        return Downsampling


class StartBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in + 3, d_out),
            nn.BatchNorm1d(d_out),
            nn.LeakyReLU(0.2),
        )

    def execute(self, p, x, o):
        x = self.mlp(p)  # [N_total, d_out]
        return p, x, o


class Downsampling(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.nsample = nsample
        self.stride = stride
        self.mre = MRE(d_in, d_out)

    def execute(self, p, x, o):
        # MRE
        x = self.mre(p, x, o)

        # fps
        count = int(o[0].item()) * self.stride // (self.stride + 1)
        n_o = [count * (i + 1) for i in range(o.shape[0])]
        n_o = jt.array(n_o, dtype='int32')
        idx = pointops.furthestsampling(p, o, n_o)  # (m,)

        n_p = p[idx]  # (m, 3)
        n_x = x[idx]  # (m, c)

        return n_p, n_x, n_o


class RFE(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.BiMLP = nn.Sequential(
            nn.Conv1d(10, 2 * d_out, 1),
            nn.BatchNorm1d(2 * d_out),
            nn.ReLU(),
            nn.Conv1d(2 * d_out, d_out, 1),
        )

        self.score_fn = nn.Sequential(
            nn.Linear(d_out * 2, d_out * 2, bias=False),
            nn.Softmax(dim=-2),
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(d_out * 2, d_out), nn.BatchNorm1d(d_out), nn.ReLU()
        )

    def execute(self, p, x):
        # position embedding
        extended_coords = p[:, 0:1, :].broadcast((p.shape[0], p.shape[1], p.shape[2]))
        neighbors = p
        dist = jt.sqrt(((extended_coords - neighbors) ** 2).sum(dim=2, keepdims=True))
        concat = jt.concat(
            [extended_coords, neighbors, extended_coords - neighbors, dist], dim=-1
        )  # [m, 16, 10]
        p_c = self.BiMLP(concat.permute(0, 2, 1)).permute(0, 2, 1)

        p_x = jt.concat([p_c, x], dim=-1)  # [m, 16, d_out*2]
        scores = self.score_fn(p_x)
        features = jt.sum(scores * p_x, dim=1, keepdims=True)
        features = self.mlp_out(features.squeeze(1))
        return features


class MRE(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.mlp0 = nn.Sequential(
            nn.Linear(d_in, d_out // 2),
            nn.BatchNorm1d(d_out // 2),
            nn.ReLU(),
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(d_out, d_out), nn.BatchNorm1d(d_out), nn.ReLU()
        )
        self.mlp01 = nn.Sequential(
            nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU()
        )
        self.Rfe_1 = RFE(d_out // 2, d_out // 2)
        self.Rfe_2 = RFE(d_out // 2, d_out // 2)

    def execute(self, p, x, o):
        x_start = x
        x = self.mlp0(x)
        xr, _ = pointops.queryandgroup(
            16, p, p, x, None, offset=o, new_offset=o, use_xyz=True, return_index=True
        )
        x = self.Rfe_1(xr[:, :, :3], xr[:, :, 3:])
        x_middle = x
        xr, _ = pointops.queryandgroup(
            16, p, p, x, None, offset=o, new_offset=o, use_xyz=True, return_index=True
        )
        x = self.Rfe_2(xr[:, :, :3], xr[:, :, 3:])
        x = jt.concat([x_middle, x], dim=1)
        x = self.mlp01(x_start) + self.mlp1(x)
        return x


def gather_neighbors(x, idx):
    """x: (B, N, C), idx: (B, N, k) local indices into the N dim -> (B, N, k, C)"""
    B, N, C = x.shape
    k = idx.shape[-1]
    batch_idx = jt.arange(B).reshape(B, 1, 1).broadcast((B, N, k))
    return x[batch_idx, idx]


class CrossAttentionPointTransformerLayer(nn.Module):
    def __init__(self, dim, attn_mlp_hidden_mult=4, num_neighbors=None):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.attn_bimlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.BatchNorm1d(dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def execute(self, x_e, x_r, x_d, pos):
        """
        x_e: (B, N, C1)
        x_r: (B, N, C2)
        x_d: (B, N, C2)
        pos: (B, N, 3)
        """
        n = x_e.shape[1]
        q = self.to_q(x_e)
        k = self.to_k(x_r)
        v = self.to_v(x_d)

        if self.num_neighbors is not None and self.num_neighbors < n:
            rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, N, N, 3)
            rel_dist = jt.sqrt((rel_pos ** 2).sum(dim=-1) + 1e-12)  # (B, N, N)
            order = jt.argsort(rel_dist, dim=-1)[0]
            indices = order[:, :, :self.num_neighbors]  # (B, N, k) smallest distances

            v = gather_neighbors(v, indices)          # (B, N, k, C)
            k = gather_neighbors(k, indices)           # (B, N, k, C)
            qk_rel = q.unsqueeze(2) - k                # (B, N, k, C)
            x_e_g = gather_neighbors(x_e, indices)     # (B, N, k, C)
        else:
            qk_rel = q.unsqueeze(2) - k.unsqueeze(1)
            x_e_g = x_e.unsqueeze(2).broadcast(qk_rel.shape)

        v = v + x_e_g
        B, N, neigh_num, C = qk_rel.shape

        sim = self.attn_bimlp((qk_rel + x_e_g).reshape(B * N * neigh_num, C))
        sim = sim.reshape(B, N, neigh_num, C)
        attn = nn.softmax(sim, dim=-2)  # softmax over the neighbor axis

        agg = jt.sum(attn * v, dim=2)  # (B, N, C)
        return agg


class Upsampling(nn.Module):
    def __init__(self, d_in_sparse_fusion, d_out, nsample):
        super().__init__()
        d_in_sparse, d_in_dense = d_in_sparse_fusion
        self.nsample = nsample
        self.d_out = d_out

        self.CrossPT_func = CrossAttentionPointTransformerLayer(
            dim=d_in_sparse,
            attn_mlp_hidden_mult=1,
            num_neighbors=16,
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_in_sparse + d_in_dense, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(),
        )

        self.dense_mlp = nn.Sequential(
            nn.Linear(d_in_dense, d_in_sparse),
            nn.BatchNorm1d(d_in_sparse),
            nn.ReLU(),
        )

    def execute(self, p1, x1, o1, p2, x2, o2, batch_size=5):
        """
        pxo1: dense
        pxo2: sparse
        """
        num_points = p1.shape[0] // batch_size

        x1_dense = self.dense_mlp(x1)

        x2_interpolated = pointops.interpolation(p2, p1, x2, o2, o1, k=8)

        x1_enhance = self.CrossPT_func(
            x1_dense.reshape(batch_size, num_points, -1),
            x2_interpolated.reshape(batch_size, num_points, -1),
            x2_interpolated.reshape(batch_size, num_points, -1),
            p1.reshape(batch_size, num_points, -1),
        ).reshape(batch_size * num_points, -1)

        x = self.mlp(jt.concat([x1_enhance, x1], dim=-1))
        return p1, x, o1
