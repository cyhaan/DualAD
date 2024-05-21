import torch
import torch.nn.functional as F
from einops import rearrange

from models.initializer import initialize_from_cfg
from torch import Tensor, nn


class MGCFR(nn.Module):
    def __init__(self, inplanes, instrides, feature_size, feature_noise, neighbor_mask, proj_mlp_ratio, memory_module,
                 hidden_dim, nhead, dim_feedforward, dropout, num_encoder_layers, num_decoder_layers, pos_embed_type,
                 activation, predict, initializer, ):

        super().__init__()
        assert isinstance(inplanes, list)
        assert isinstance(instrides, list) and len(instrides) == 1
        self.inplanes = inplanes
        self.instrides = instrides
        self.feature_size = feature_size
        self.feature_noise = feature_noise
        self.predict = predict
        self.embed_dim = hidden_dim
        self.neighbor_mask = neighbor_mask

        abs_pos_embed = PositionEmbeddingLearned(
            feature_size, hidden_dim) if pos_embed_type == 'learned' else None

        in_dims = sum(inplanes)
        out_dims = hidden_dim * num_decoder_layers
        hid_dims = int((in_dims + out_dims) * proj_mlp_ratio)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dims, hid_dims),
            _get_activation_layer(activation),
            nn.Linear(hid_dims, out_dims)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(out_dims, hid_dims),
            _get_activation_layer(activation),
            nn.Linear(hid_dims, in_dims)
        )
        # Memory Module
        self.memory_modules = nn.ModuleList([
            MemoryModule(**memory_module) for _ in range(num_decoder_layers)
        ])
        # Encoder
        self.encoder = TransformerEncoder(
            num_encoder_layers,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            abs_pos_embed=abs_pos_embed,
            activation=activation,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers,
            hidden_dim=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            abs_pos_embed=abs_pos_embed,
            activation=activation,
            dropout=dropout
        )
        initialize_from_cfg(self, initializer)

    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .cuda()
        )
        return mask

    def to_anomaly(self, x):
        # x : (B,N,D)
        std = x.var(dim=1, keepdim=True).sqrt()
        noise = torch.randn(x.shape, device=x.device) * std
        scale = self.feature_noise.scale
        noise = noise * scale
        x = x + noise
        return x

    def to_memories(self, x):
        x = self.in_proj(x)
        # Split heads
        feats = list(torch.split(x, self.embed_dim, dim=-1))
        # Memory addressing
        outs = [block(feat) for feat, block in zip(feats, self.memory_modules)]
        attn_weights, attn_weights_max, memories_attn, memories_qtz = [list(item) for item in zip(*outs)]
        return {
            "feats_proj": feats,
            "attn_weights": attn_weights,
            "attn_weights_max": attn_weights_max,
            "memories_attn": memories_attn,
            "memories_qtz": memories_qtz
        }

    def cal_anomaly_score(self, feature_align, feature_rec):
        feats_align = feature_align.split(self.inplanes, dim=1)
        feats_rec = feature_rec.split(self.inplanes, dim=1)
        a_maps = [torch.mean((fr - fa)**2, dim=1, keepdim=True).sqrt() for fa, fr in zip(feats_align, feats_rec)]
        anomaly_map = torch.cat(a_maps, dim=1).prod(dim=1, keepdim=True)  # B X 1 X H X W
        pred_pixel = F.interpolate(anomaly_map, scale_factor=self.instrides[0], mode='bilinear', align_corners=True)
        pred_image = F.avg_pool2d(pred_pixel, self.predict.avgpool_size, stride=1)
        pred_image = pred_image.flatten(1).max(axis=1)[0]
        return pred_pixel, pred_image


    def forward(self, input):
        feature_align = input["feature_align"]
        x = rearrange(
            feature_align, "b c h w -> b (h w) c"
        )
        mem_outs = self.to_memories(x)
        mems_input = mem_outs
        mems_anomal = None
        if self.training and self.feature_noise:
            feature_anomal = self.to_anomaly(x)
            mem_outs = self.to_memories(feature_anomal)
            mems_anomal = mem_outs
        if self.neighbor_mask:
            mask = self.generate_mask(
                self.feature_size, self.neighbor_mask.neighbor_size
            )
            mask_enc = mask if self.neighbor_mask.mask[0] else None
            mask_dec1 = mask if self.neighbor_mask.mask[1] else None
            mask_dec2 = mask if self.neighbor_mask.mask[2] else None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None
        memories = mem_outs["memories_attn"]
        enc_out = self.encoder(sum(memories), mask_enc)
        dec_outs = self.decoder(memories, enc_out, mask_dec1, mask_dec2)
        x = self.out_proj(torch.cat(dec_outs, dim=-1))
        feature_rec = rearrange(x, " b (h w) d -> b d h w ", h=self.feature_size[0])

        pred_pixel = None
        pred_image = None
        if not self.training:
            pred_pixel, pred_image = self.cal_anomaly_score(feature_align, feature_rec)  # B x 1 x H x W
        return {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "mems_input": mems_input,
            "mems_anomal": mems_anomal,
            "pred_pixel": pred_pixel,
            "pred_image": pred_image,
            "outplanes": self.inplanes
        }


class MemoryModule(nn.Module):
    def __init__(self, num_embeds, embed_dim, init_norm=1, scale=None, shrink_thres=None):
        super().__init__()
        self.num_embeds = num_embeds
        self.embed_dim = embed_dim
        self.scale = scale
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.Tensor(num_embeds, embed_dim))  # M x C
        self.shrink_thres = shrink_thres if shrink_thres is not None else 1 / self.num_embeds

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        with torch.no_grad():
            norm = self.weight.norm(dim=-1, keepdim=True)/self.init_norm
            self.weight.div_(norm)

    def forward(self, x: Tensor):
        # x: B x N x C
        # m: M x C
        attn_weight = F.linear(F.normalize(x, dim=-1)*self.scale, F.normalize(self.weight, dim=-1))  # B x N x M

        attn_weight = F.softmax(attn_weight, dim=-1)
        attn_weight = F.threshold(attn_weight, self.shrink_thres, 0)
        attn_weight = F.normalize(attn_weight, p=1, dim=-1)

        attn_out = F.linear(attn_weight, self.weight.transpose(0, 1))
        attn_weight_max, slices = attn_weight.max(dim=-1, keepdim=True)
        qtz_out = self.weight[slices.flatten()].view(x.shape)

        return attn_weight, attn_weight_max, attn_out, qtz_out

    def extra_repr(self) -> Tensor:
        return 'num_embeds={num_embeds}, embed_dim={embed_dim}'.format(**self.__dict__)


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(**kwargs) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(**kwargs) for _ in range(num_layers)
        ])

    def forward(self, feats, enc_out, trg_mask=None, src_mask=None):
        outputs = []
        x = enc_out
        for feat, layer in zip(feats, self.layers):
            x = layer(feat, x, enc_out, trg_mask, src_mask)
            outputs.append(x)
        return outputs


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward, abs_pos_embed=None, activation="relu", dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_dim, nhead, dropout=dropout)
        self.ffn = FeedForward(hidden_dim, hidden_dim, dim_feedforward, activation, dropout=dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.abs_pos_embed = abs_pos_embed or nn.Identity()

    def forward(self, x, mask=None):
        shortcut = x
        q = k = self.abs_pos_embed(x)
        x = self.attn(q, k, value=x, mask=mask)
        x = x + shortcut
        x = self.norm1(x)
        shortcut = x
        x = self.ffn(x)
        x = x + shortcut
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, nhead, dim_feedforward, abs_pos_embed=None, activation="relu", dropout=0.1):
        super().__init__()
        self.attn1 = MultiHeadAttention(hidden_dim, nhead, dropout=dropout)
        self.attn2 = MultiHeadAttention(hidden_dim, nhead, dropout=dropout)
        self.ffn = FeedForward(hidden_dim, hidden_dim, dim_feedforward, activation, dropout=dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.abs_pos_embed = abs_pos_embed or nn.Identity()

    def forward(self, x, dec_out, enc_out, trg_mask=None, src_mask=None):
        shortcut = x
        q = self.abs_pos_embed(x)
        k = self.abs_pos_embed(enc_out)
        x = self.attn1(q, k, value=enc_out, mask=src_mask)
        x = x + shortcut
        x = self.norm1(x)
        shortcut = x
        q = self.abs_pos_embed(x)
        k = self.abs_pos_embed(dec_out)
        x = self.attn2(q, k, value=dec_out, mask=trg_mask)
        x = x + shortcut
        x = self.norm2(x)
        shortcut = x
        x = self.ffn(x)
        x = x + shortcut
        x = self.norm3(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., qk_scale=None, qkv_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def split_head(self, x):
        return rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)

    def forward(self, query, key, value, mask=None):
        # Compute q, k, v
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        # Split head
        q, k, v = self.split_head(q), self.split_head(k), self.split_head(v)

        q = q * self.scale
        # [B, Nh, HW, HW]
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)

        # (B, Nh, dvh, HW)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # (B,dim, H, W)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, activation="relu", dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.act = _get_activation_layer(activation)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, embed_dim):
        super().__init__()

        assert embed_dim % 2 == 0

        self.feature_size = feature_size  # H, W
        self.embed_dim = embed_dim
        self.row_embed = nn.Parameter(torch.Tensor(self.feature_size[0], self.embed_dim // 2))  # H x C
        self.col_embed = nn.Parameter(torch.Tensor(self.feature_size[1], self.embed_dim // 2))  # W x C
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed)
        nn.init.uniform_(self.col_embed)

    def forward(self, x):
        pos = torch.cat([
            torch.stack([self.col_embed] * self.feature_size[0], dim=0),  # H x W x C // 2
            torch.stack([self.row_embed] * self.feature_size[1], dim=1)  # H x W x C // 2
        ], dim=-1).flatten(0, 1).unsqueeze(0)  # 1 x (H x W) x (C // 2 x 2)
        x = x + pos
        return x

    def extra_repr(self) -> Tensor:
        return 'feature_size={feature_size}, embed_dim={embed_dim}'.format(**self.__dict__)


def _get_activation_layer(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
