"""
Unimodal transformer encoder (Audio-only or Vision-only pretraining)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

# re-use your utilities
from MMSA.transformations.pool.mmaug import SoftPerm_Fast
from MMSA.models.subNets.transformers_encoder.transformer import SinusoidalPositionalEmbedding


class Masking(nn.Module):
    """Unimodal time-step masking."""
    def __init__(self, p: float):
        super().__init__()
        assert 0.0 <= p <= 1.0
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.0:
            B, L, _ = x.shape
            mask = (torch.rand(B, L, device=x.device) > self.p).unsqueeze_(2)  # (B,L,1)
            x = x * mask
        return x


class UNI_ENC(nn.Module):
    """
    Unimodal encoder that operates on a single modality chosen by args['uni_enc']['modality'].

    Expects the SAME call signature as BI_ENC:
        forward(self, _, x_a, x_v)

    but will internally pick just one of (x_a or x_v).
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        # ---- where we read config from ----
        cfg_uni = args.get("unienc", {})

        # modality to train: 'audio' or 'vision'
        self.modality: str = cfg_uni.get("modality", "audio").lower()
        assert self.modality in ("audio", "vision"), "uni_enc.modality must be 'audio' or 'vision'"

        # in-shape meta
        # args.feature_dims = (d_text, d_audio, d_vision)
        _, self.orig_d_a, self.orig_d_v = args.feature_dims
        # args.seq_lens = (L_text, L_audio, L_vision)
        _, self.l_a, self.l_v = args.seq_lens

        # hyperparams (fall back to av_enc if not provided in uni_enc)
        self.d_enc       = cfg_uni.get("d_enc", 16)
        self.n_head      = cfg_uni.get("n_head", 4)
        self.nlevels     = cfg_uni.get("nlevels", 3)
        self.maxlen      = cfg_uni.get("maxlen", 39)
        self.enc_dropout = cfg_uni.get("enc_dropout", 0.1)
        self.use_ln      = cfg_uni.get("use_ln", False)
        self.use_bn      = cfg_uni.get("use_bn", True)
        self.use_softperm= cfg_uni.get("use_softperm", True)
        self.p_perm      = cfg_uni.get("p_perm", 0.2)
        self.use_masking = cfg_uni.get("use_m3", False)
        self.p_mask      = cfg_uni.get("p_mask", 0.1)
        self.pooling = cfg_uni.get("pooling", "mean")

        # choose original dims / seq lens by modality
        if self.modality == "audio":
            self.orig_d_m = self.orig_d_a
            self.seq_len  = self.l_a
        else:
            self.orig_d_m = self.orig_d_v
            self.seq_len  = self.l_v

        # --------- Normalization ---------
        if self.use_ln:
            self.LN = nn.LayerNorm(self.orig_d_m)
        else:
            self.LN = nn.Identity()

        if self.use_bn:
            # BN over features (treat time as "length" -> BN1d expects [B, C, L])
            self.BN = nn.BatchNorm1d(self.orig_d_m)
        else:
            self.BN = nn.Identity()

        # --------- Positional + projection ---------
        self.proj = nn.Linear(self.orig_d_m, self.d_enc, bias=False)
        self.embed_scale = math.sqrt(self.d_enc)
        self.embed_positions = SinusoidalPositionalEmbedding(self.d_enc)

        # --------- Soft permutation (sequence augmentation) ---------
        if self.use_softperm:
            self.sperm = SoftPerm_Fast(p_feat=self.p_perm, maxlen=self.seq_len)

        # --------- Masking ---------
        self.mask = Masking(self.p_mask)

        # --------- Encoder ---------
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_enc,
                nhead=self.n_head,
                dropout=self.enc_dropout,
                dim_feedforward=4 * self.d_enc,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ),
            num_layers=self.nlevels
        )

        # --------- Head ---------
        self.head = nn.Linear(self.d_enc, 1)

    # ------- helpers -------
    def _align(self, x: torch.Tensor) -> torch.Tensor:
        """
        Align the raw sequence to maxlen by chunk-mean pooling like your BI_ENC.align
        """
        raw_seq_len = x.size(1)
        if raw_seq_len == self.maxlen:
            return x
        if raw_seq_len // self.maxlen == raw_seq_len / self.maxlen:
            pad_len = 0
            pool_size = raw_seq_len // self.maxlen
        else:
            pad_len = self.maxlen - raw_seq_len % self.maxlen
            pool_size = raw_seq_len // self.maxlen + 1
        pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
        x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.maxlen, -1)
        x = x.mean(dim=1)
        return x

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return h.mean(dim=1)
        return h[:, -1, :]  # 'last'

    # ------- forward -------
    def forward(self, _, x_a: torch.Tensor, x_v: torch.Tensor) -> torch.Tensor:
        """
        Keep same signature as BI_ENC: (dummy_text, audio, vision)
        but only use the selected modality.
        Returns logits: (B, C)
        """
        x = x_a if self.modality == "audio" else x_v  # (B, L, D_orig)

        # LayerNorm on inputs (feature dim)
        x = self.LN(x)

        # Soft-perm (sequence augmentation)
        if self.training and self.use_softperm:
            x = self.sperm(x)

        # BN (across feature dim per timestep sequence) + align to maxlen
        x = self._align(x)
        if isinstance(self.BN, nn.BatchNorm1d):
            x = self.BN(x.transpose(1, 2)).transpose(1, 2)  # (B,L,D) -> BN -> (B,L,D)

        # Project + scale
        x = self.proj(x) * self.embed_scale

        # Positional embedding + dropout
        x = x + self.embed_positions(x[:, :, 0])
        x = F.dropout(x, p=self.enc_dropout, training=self.training)

        # Time-step masking
        if self.training and self.use_masking and self.p_mask > 0.0:
            x = self.mask(x)

        # Encode
        h = self.enc(x)  # (B, L, d_enc)

        # Pooling to a single vector
        h = self._pool(h)  # (B, d_enc)

        # Logits
        y = self.head(h)   # (B, C)
        return y
