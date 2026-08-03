import math
import jittor as jt
import jittor.nn as nn

from .blocks import block_decider
from .classify import Classify
from .fusion import FusionHead


class FeatureExtraction(nn.Module):
    def __init__(self, d_in=0, d_out=32,
                 n_cls=3, nsample=16, stride_list=[4, 3, 2, 1],
                 architecture=None,
                 classify_ckpt=None, classify_frame_knn=32):
        super().__init__()
        architecture = ['startblock',
                        'downsample',
                        'downsample',
                        'downsample',
                        'downsample',
                        'upsample',
                        'upsample',
                        'upsample',
                        'upsample', ]
        d_in = d_in
        d_out = d_out
        n_cls = n_cls
        nsample = nsample
        stride_list = stride_list
        stride_dim_list = [1.5, 1.5, 1.5, 1.5]
        stride = 1
        stride_idx = 0
        d_prev = d_in

        # construct encoder
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []

        for block_name in architecture:
            if 'downsample' in block_name:
                self.encoder_skip_dims.append(d_prev)
                stride = stride_list[stride_idx]
                d_out = int(d_out * stride_dim_list[stride_idx])
                stride_idx += 1
                self.encoder_blocks.append(
                    block_decider(block_name)(d_prev, d_out, nsample, stride)
                )
            elif 'upsample' in block_name:
                skip_dim = self.encoder_skip_dims.pop()
                d_out = skip_dim
                self.decoder_blocks.append(
                    block_decider(block_name)([d_prev, skip_dim], d_out, nsample)
                )
            else:
                self.encoder_blocks.append(
                    block_decider(block_name)(d_prev, d_out, nsample, stride)
                )
            d_prev = d_out

        self.linear0_1 = nn.Linear(d_out, 128, bias=False)
        self.linear0_2 = nn.Linear(128, 64)
        self.linear0_3 = nn.Linear(64, n_cls)
        self.fusion_head = FusionHead(feat_dim=d_out)
        self.use_fusion = False

        # Competition rule: no data/weights from outside the provided dataset.
        # So we never load the original torch-format pretrained/classify.ckpt
        # (it was trained on ASDN's own paper data, not this competition's
        # ShapeNet set). Instead: build a fresh Classify, and if a Jittor-
        # native checkpoint (trained from scratch on THIS competition's data
        # via train_classifier_on_starter.py) is given, load that.
        self.classify = Classify(frame_knn=classify_frame_knn)
        if classify_ckpt is not None:
            self.classify.load(classify_ckpt)
            self.classify.eval()

    def execute(self, p, x, o, feat_T=None, t_frac=None, return_feat=False):
        flag = False
        lambda_layer = 4
        L = 4
        gamma = 0.396

        def assign_n_layer_based_on_rho(rho_list):
            return [math.ceil(L - (L - 1) * math.log(gamma * float(rho) + 1)) for rho in rho_list]

        batch_size = p.shape[0]
        num_points = p.shape[1]
        p_from_encoder = []
        x_from_encoder = []
        o_from_encoder = []
        x_out = []

        rho_list, _ = self.classify.feature_nets[0](p.reshape(batch_size, num_points, -1), None)
        n_layers = assign_n_layer_based_on_rho(rho_list)
        layer_2 = [i for i, n in enumerate(n_layers) if n == 2]
        layer_3 = [i for i, n in enumerate(n_layers) if n == 3]
        layer_4 = [i for i, n in enumerate(n_layers) if n == 4]

        # encoder
        for block_i, block in enumerate(self.encoder_blocks):

            if block_i == 3 and layer_2:
                if layer_3 or layer_4:
                    p = p.reshape(batch_size, int(o[0].item()), -1)[layer_3 + layer_4, :, :].reshape(-1, p.shape[-1])
                    x = x.reshape(batch_size, int(o[0].item()), -1)[layer_3 + layer_4, :, :].reshape(-1, x.shape[-1])
                    o = o[:len(layer_3 + layer_4)]
                else:
                    flag = True

            if block_i == 4 and layer_3:
                if layer_4:
                    if layer_2:
                        p = p[len(layer_3) * int(o[0].item()):, :]
                        x = x[len(layer_3) * int(o[0].item()):, :]
                        o = o[:len(layer_4)]
                    else:
                        p = p.reshape(batch_size, int(o[0].item()), -1)[layer_4, :, :].reshape(-1, p.shape[-1])
                        x = x.reshape(batch_size, int(o[0].item()), -1)[layer_4, :, :].reshape(-1, x.shape[-1])
                        o = o[:len(layer_4)]
                else:
                    flag = True

            if not flag:
                p, x, o = block(p.reshape(-1, 3), x, o)
                p_from_encoder.append(p.reshape(-1, 3))
                x_from_encoder.append(x)
                o_from_encoder.append(o)
            else:
                lambda_layer = block_i - 1
                break

        x_dense = x_from_encoder.pop()
        p_dense = p_from_encoder.pop()
        o_dense = o_from_encoder.pop()

        for block_i, block in enumerate(self.decoder_blocks[L - lambda_layer:]):
            x_dense = x_from_encoder.pop()
            p_dense = p_from_encoder.pop()
            o_dense = o_from_encoder.pop()

            p_dense_corres = p_dense
            x_dense_corres = x_dense
            o_dense_corres = o_dense

            if block_i == 0 - (L - lambda_layer) and layer_3:
                if layer_2:
                    p_dense_corres = p_dense[len(layer_3) * int(o_dense[0].item()):, :]
                    x_dense_corres = x_dense[len(layer_3) * int(o_dense[0].item()):, :]
                    o_dense_corres = o_dense[:len(layer_4)]
                else:
                    p_dense_corres = p_dense.reshape(batch_size, int(o_dense[0].item()), -1)[layer_4, :, :].reshape(-1, p_dense.shape[-1])
                    x_dense_corres = x_dense.reshape(batch_size, int(o_dense[0].item()), -1)[layer_4, :, :].reshape(-1, x_dense.shape[-1])
                    o_dense_corres = o_dense[:len(layer_4)]

            if block_i == 1 - (L - lambda_layer) and layer_2:
                p_dense_corres = p_dense.reshape(batch_size, int(o_dense[0].item()), -1)[layer_3 + layer_4, :, :]
                x_dense_corres = x_dense.reshape(batch_size, int(o_dense[0].item()), -1)[layer_3 + layer_4, :, :]
                o_dense_corres = o_dense[:len(layer_3 + layer_4)]

            p, x, o = block(p_dense_corres, x_dense_corres, o_dense_corres, p.reshape(-1, 3), x, o, o_dense_corres.shape[0])

            if p.shape[0] != p_dense.shape[0]:
                if block_i == 0 - (L - lambda_layer):
                    if layer_2:
                        p_dense[len(layer_3) * int(o_dense[0].item()):, :] = p
                        x_dense[len(layer_3) * int(o_dense[0].item()):, :] = x
                        o = o_dense
                        p = p_dense
                        x = x_dense
                    else:
                        p_dense.reshape(batch_size, int(o_dense[0].item()), -1)[layer_4, :, :] = p.reshape(len(layer_4), -1, p.shape[-1])
                        x_dense.reshape(batch_size, int(o_dense[0].item()), -1)[layer_4, :, :] = x.reshape(len(layer_4), -1, x.shape[-1])
                        o = o_dense
                        p = p_dense.reshape(-1, p.shape[-1])
                        x = x_dense.reshape(-1, x.shape[-1])

                if block_i == 1 - (L - lambda_layer):
                    p_dense.reshape(batch_size, int(o_dense[0].item()), -1)[layer_3 + layer_4, :, :] = p
                    x_dense.reshape(batch_size, int(o_dense[0].item()), -1)[layer_3 + layer_4, :, :] = x
                    o = o_dense
                    p = p_dense.reshape(-1, p.shape[-1])
                    x = x_dense.reshape(-1, x.shape[-1])
                    
        
        if self.use_fusion:
            feat_t = x.reshape(batch_size, num_points, -1)   # E(x^t)
            pos = p.reshape(batch_size, num_points, -1)      # current positions
            if t_frac is None:
                t_frac = jt.zeros((batch_size, num_points, 1))
            feat_T_used = feat_t if feat_T is None else feat_T
            x_out = self.fusion_head(pos, t_frac, feat_t, feat_T_used)
            if return_feat:
                return x_out, feat_t
        else:
            x = nn.relu(self.linear0_1(x))
            x = nn.relu(self.linear0_2(x))
            x_out = jt.tanh(self.linear0_3(x))
            x_out = x_out.reshape(batch_size, num_points, -1)

        return x_out
