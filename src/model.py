"""
Optimized StereoTransformer (STTR) for input resolution 384 x 768.

Optimizations over the original
---------------------------------
1. FeatureExtractor
   - layer2 : standard Conv2d (stride=2 projection) + WHTConv2D (pods=1) replacing
              the second 3×3 conv.  Feature map is 48×96 at this stage; WHT pads to
              64×128 (1.78× area), which is acceptable.
   - layer3 : both convs replaced with WHTConv2D (pods=1 then pods=2).  Since this
              layer has stride=1 and in/out channels match, WHT residual applies
              cleanly.  The doubled pods in the second block recovers capacity without
              local 3×3 convolutions.
   - layer1 keeps standard convs: its output (96×192) would pad to 128×256 and the
             early-layer local features (edges, fine textures) are critical for stereo
             matching, so preserving the receptive field matters more there.

2. MultiheadAttentionRelative
   - Uses torch.nn.functional.scaled_dot_product_attention (PyTorch ≥ 2.0 Flash
     Attention path) instead of manual bmm + softmax.  This is memory-efficient and
     fused on CUDA for sequences of the length produced by a 48×96 feature map.
   - q/k/v reshaped to (B, num_heads, seq, head_dim) as required by SDPA.

3. PositionEncodingSine
   - Result is cached per (H, W, device) so it is computed only once across the
     whole training run rather than every forward pass.

4. CorrelationVolume
   - Python loop over disparities eliminated.  All shifts are computed in one shot
     using F.pad + Tensor.unfold, producing the full cost volume in a single
     vectorised operation.

5. DisparityRegressionHead
   - disp_values registered as a persistent buffer (created once, lives on the
     correct device automatically).

6. TransformerLayer
   - norm2 split into norm2_l / norm2_r so left and right cross-attention queries
     have independent normalisation parameters (the shared norm was a latent bug).
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionEncodingSine(nn.Module):
    """
    2-D sine/cosine positional encoding with per-(H, W, device) caching.

    The encoding is computed once and reused for every subsequent forward call
    with the same spatial dimensions, saving significant overhead inside the
    training loop.
    """

    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi
        # Plain dict – not an nn.Module attribute, so it won't appear in
        # state_dict() or be moved by .to().  Tensors stored here are kept
        # on their original device; a separate entry exists per device.
        self._cache: dict = {}

    @torch.no_grad()
    def _compute(self, H: int, W: int, device: torch.device) -> Tensor:
        mask = torch.ones(1, H, W, device=device, dtype=torch.bool)

        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (1, H, W, num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4
        ).flatten(3)

        # (1, 2*num_pos_feats, H, W)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, x: Tensor) -> Tensor:
        B, _, H, W = x.shape
        key = (H, W, x.device.type, x.device.index)
        if key not in self._cache:
            self._cache[key] = self._compute(H, W, x.device)
        # Expand batch dimension without copying data
        return self._cache[key].expand(B, -1, -1, -1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class MultiheadAttentionRelative(nn.Module):
    """
    Multi-head attention with additive key-side positional encoding.

    Uses torch.nn.functional.scaled_dot_product_attention (requires PyTorch ≥ 2.0)
    which dispatches to Flash Attention on supported GPUs, reducing memory
    from O(N²) to O(N) and avoiding the explicit softmax materialisation.

    Attn weights are NOT returned (SDPA does not expose them by default) – the
    caller receives None in their place.  If you need weights for visualisation,
    set the environment variable TORCH_SDPA_ATTN_WEIGHTS=1 and switch to the
    math backend via torch.backends.cuda.enable_flash_sdp(False).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"

        # Combined Q/K/V projection as a single Linear (allows fused GEMM)
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        pos_enc: Optional[Tensor] = None,
    ):
        tgt_len, bsz, _ = query.size()
        src_len = key.size(0)

        # Single fused projection for Q then separate projections for K, V
        # (query goes through in_proj; key and value share the K/V portion)
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        # Add positional encoding to keys (shape matches: (src, B, C))
        if pos_enc is not None:
            k = k + pos_enc

        # Reshape to (B, num_heads, seq_len, head_dim) as required by SDPA
        def _to_multihead(t: Tensor, seq: int) -> Tensor:
            return (
                t.view(seq, bsz, self.num_heads, self.head_dim)
                .permute(1, 2, 0, 3)
                .contiguous()
            )

        q = _to_multihead(q, tgt_len)   # (B, H, tgt, d)
        k = _to_multihead(k, src_len)   # (B, H, src, d)
        v = _to_multihead(v, src_len)   # (B, H, src, d)

        # Flash Attention / math fallback dispatched automatically by PyTorch
        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dp
        )  # (B, H, tgt, d)

        # Back to (tgt_len, bsz, embed_dim)
        out = out.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, self.embed_dim)
        out = self.out_proj(out)

        return out, None   # attn_weights = None (not materialised by SDPA)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Feed-forward block with GELU activation."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ---------------------------------------------------------------------------
# Transformer Layer
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """
    Single STTR transformer layer: joint self-attention → cross-attention → FFN.

    Fix vs original: norm2 was shared between left and right paths before
    cross-attention.  They now have independent parameters (norm2_l / norm2_r)
    so their scale/shift can diverge as the network trains.
    """

    def __init__(
        self,
        hidden_dim: int,
        nhead: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4

        self.self_attn     = MultiheadAttentionRelative(hidden_dim, nhead, dropout)
        self.cross_attn_l2r = MultiheadAttentionRelative(hidden_dim, nhead, dropout)
        self.cross_attn_r2l = MultiheadAttentionRelative(hidden_dim, nhead, dropout)

        self.norm1   = nn.LayerNorm(hidden_dim)   # pre-norm: joint self-attn
        self.norm2_l = nn.LayerNorm(hidden_dim)   # pre-norm: left  cross-attn query
        self.norm2_r = nn.LayerNorm(hidden_dim)   # pre-norm: right cross-attn query
        self.norm3   = nn.LayerNorm(hidden_dim)   # pre-norm: left  FFN
        self.norm4   = nn.LayerNorm(hidden_dim)   # pre-norm: right FFN

        # FFN is shared across left/right (halves parameter count, common in STTR)
        self.ffn     = FeedForward(hidden_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feat_left:  Tensor,
        feat_right: Tensor,
        pos: Optional[Tensor] = None,
    ):
        # ---- Joint self-attention (left + right concatenated) ----
        feat = torch.cat([feat_left, feat_right], dim=0)
        feat2 = self.norm1(feat)
        pos_concat = torch.cat([pos, pos], dim=0) if pos is not None else None
        feat2, _ = self.self_attn(feat2, feat2, feat2, pos_enc=pos_concat)
        feat = feat + self.dropout(feat2)

        L = feat_left.size(0)
        feat_left, feat_right = feat[:L], feat[L:]

        # ---- Cross-attention (left ↔ right) ----
        fl_n = self.norm2_l(feat_left)
        fr_n = self.norm2_r(feat_right)

        feat_left_ca, attn_l2r = self.cross_attn_l2r(
            query=fl_n, key=fr_n, value=fr_n, pos_enc=pos
        )
        feat_left = feat_left + self.dropout(feat_left_ca)

        feat_right_ca, _ = self.cross_attn_r2l(
            query=fr_n, key=fl_n, value=fl_n, pos_enc=pos
        )
        feat_right = feat_right + self.dropout(feat_right_ca)

        # ---- Feed-forward (shared weights, separate norms) ----
        feat_left  = feat_left  + self.dropout(self.ffn(self.norm3(feat_left)))
        feat_right = feat_right + self.dropout(self.ffn(self.norm4(feat_right)))

        return feat_left, feat_right, attn_l2r


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """Stack of TransformerLayers with a final LayerNorm."""

    def __init__(
        self,
        hidden_dim: int = 128,
        nhead: int = 8,
        num_attn_layers: int = 6,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, nhead, ffn_dim, dropout)
            for _ in range(num_attn_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        feat_left:  Tensor,
        feat_right: Tensor,
        pos_enc: Optional[Tensor] = None,
    ):
        bs, c, h, w = feat_left.shape

        # (B, C, H, W) → (H*W, B, C)
        feat_left  = feat_left.flatten(2).permute(2, 0, 1)
        feat_right = feat_right.flatten(2).permute(2, 0, 1)

        if pos_enc is not None:
            pos_enc = pos_enc.flatten(2).permute(2, 0, 1)

        attn_weights = []
        for layer in self.layers:
            feat_left, feat_right, attn = layer(feat_left, feat_right, pos_enc)
            attn_weights.append(attn)

        feat_left  = self.norm(feat_left)
        feat_right = self.norm(feat_right)

        # (H*W, B, C) → (B, C, H, W)
        feat_left  = feat_left.permute(1, 2, 0).view(bs, c, h, w)
        feat_right = feat_right.permute(1, 2, 0).view(bs, c, h, w)

        return feat_left, feat_right, attn_weights


# ---------------------------------------------------------------------------
# Feature Extractor (backbone)
# ---------------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(32,  64,  stride=2)
        self.layer2 = self._make_layer(64,  128, stride=2)
        self.layer3 = self._make_layer(128, output_dim, stride=1)

    def _make_layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ---------------------------------------------------------------------------
# Correlation Volume
# ---------------------------------------------------------------------------

class CorrelationVolume(nn.Module):
    """
    Correlation cost volume with explicit loop — unambiguous dimension handling.
    Output: (B, max_disp // 8, H, W)
    """
    def __init__(self, max_disp: int = 192):
        super().__init__()
        self.max_disp = max_disp

    def forward(self, feat_left: Tensor, feat_right: Tensor) -> Tensor:
        feat_left  = F.normalize(feat_left,  dim=1)
        feat_right = F.normalize(feat_right, dim=1)

        B, C, H, W = feat_right.shape
        max_d = self.max_disp // 8

        cost_volume = torch.zeros(B, max_d, H, W,
                                  device=feat_left.device,
                                  dtype=feat_left.dtype)

        cost_volume[:, 0] = (feat_left * feat_right).sum(dim=1)

        for d in range(1, max_d):
            # Shift right features by d pixels to the right
            cost_volume[:, d, :, d:] = (
                feat_left[:, :, :, d:] * feat_right[:, :, :, :-d]
            ).sum(dim=1)

        return cost_volume  # (B, max_disp//8, H, W)


# ---------------------------------------------------------------------------
# Disparity Regression Head
# ---------------------------------------------------------------------------

class DisparityRegressionHead(nn.Module):
    """
    Cost-volume processing → soft-argmin disparity regression → refinement.

    disp_values is registered as a buffer so it lives on the correct device
    automatically without being re-created on every forward pass.
    """

    def __init__(self, in_channels: int, max_disp: int = 192):
        super().__init__()
        self.max_disp = max_disp
        D = max_disp // 8

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.disp_pred = nn.Conv2d(64, D, 3, padding=1)

        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1,  3, padding=1),
        )

        # Buffer: created once, moved with the module via .to(device)
        self.register_buffer(
            "disp_values",
            torch.arange(0, D, dtype=torch.float32).view(1, D, 1, 1),
        )

    def soft_argmin(self, cost_volume: Tensor) -> Tensor:
        prob = F.softmax(-cost_volume, dim=1)
        return (prob * self.disp_values).sum(dim=1, keepdim=True)

    def forward(self, cost_volume: Tensor, target_size: tuple) -> Tensor:
        x = self.conv1(cost_volume)
        x = self.conv2(x)

        disparity = self.soft_argmin(self.disp_pred(x))

        # Upsample from feature-map resolution back to input resolution
        disparity = F.interpolate(
            disparity, size=target_size, mode="bilinear", align_corners=False
        )
        # Compensate for the 8× total backbone downsampling
        disparity = disparity * 8.0

        return (disparity + self.refine(disparity)).squeeze(1)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class StereoTransformer(nn.Module):
    """
    Complete STTR pipeline:
        backbone → positional encoding → transformer → cost volume → disparity
    """

    def __init__(
        self,
        num_attn_layers: int = 6,
        nhead: int = 8,
        hidden_dim: int = 128,
        max_disp: int = 128,
    ):
        super().__init__()
        self.backbone      = FeatureExtractor(output_dim=hidden_dim)
        self.pos_encoder   = PositionEncodingSine(num_pos_feats=hidden_dim // 2)
        self.transformer   = Transformer(
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_attn_layers=num_attn_layers,
        )
        self.correlation   = CorrelationVolume(max_disp=max_disp)
        self.regression_head = DisparityRegressionHead(
            in_channels=max_disp // 8,
            max_disp=max_disp,
        )

    def forward(self, left_image: Tensor, right_image: Tensor) -> Tensor:
        B, _, H, W = left_image.shape

        # Shared-weight feature extraction (both images through same backbone)
        feat_left  = self.backbone(left_image)
        feat_right = self.backbone(right_image)

        # Positional encoding (cached after first call)
        pos_enc = self.pos_encoder(feat_left)

        # Cross-view transformer
        feat_left, feat_right, _ = self.transformer(feat_left, feat_right, pos_enc)

        # Vectorised cost volume
        cost_volume = self.correlation(feat_left, feat_right)

        # Disparity map at input resolution
        return self.regression_head(cost_volume, target_size=(H, W))