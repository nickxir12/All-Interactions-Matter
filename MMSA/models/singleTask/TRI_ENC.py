"""
Tri-modal transformer encoder that jointly trains:
1. Audio-only encoder
2. Vision-only encoder  
3. Audio-Visual bimodal encoder
"""

import math
from typing import Dict, Tuple, Optional

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


class AudioEncoder(nn.Module):
    """Audio-only encoder that reads config from tri_enc.audio_encoder."""
    
    def __init__(self, orig_dim: int, seq_len: int, tri_enc_config: Dict):
        super().__init__()
        self.orig_dim = orig_dim
        self.seq_len = seq_len
        
        # Get audio encoder specific configuration
        config = tri_enc_config.get("audio_encoder", {})
        
        # Extract audio encoder configuration with defaults
        self.d_enc = config.get("d_enc", 64)
        self.n_head = config.get("n_head", 4)
        self.nlevels = config.get("nlevels", 3)
        self.maxlen = config.get("maxlen", 50)
        self.enc_dropout = config.get("enc_dropout", 0.1)
        self.use_ln = config.get("use_ln", False)
        self.use_bn = config.get("use_bn", False)
        self.use_softperm = config.get("use_softperm", False)
        self.p_perm = config.get("p_perm", 0.0)
        self.p_mask = config.get("p_mask", 0.0)
        self.pooling = config.get("pooling", "mean")
        
        # Normalization
        self.LN = nn.LayerNorm(orig_dim) if self.use_ln else nn.Identity()
        self.BN = nn.BatchNorm1d(orig_dim) if self.use_bn else nn.Identity()
        
        # Projection and positional encoding
        self.proj = nn.Linear(orig_dim, self.d_enc, bias=False)
        self.embed_scale = math.sqrt(self.d_enc)
        self.embed_positions = SinusoidalPositionalEmbedding(self.d_enc)
        
        # Augmentation
        if self.use_softperm:
            self.sperm = SoftPerm_Fast(p_feat=self.p_perm, maxlen=seq_len)
        self.mask = Masking(self.p_mask)
        
        # Encoder
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
        
    def _align(self, x: torch.Tensor) -> torch.Tensor:
        """Align sequence to maxlen via chunk-mean pooling."""
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
        """Pool sequence to single vector."""
        if self.pooling == "mean":
            return h.mean(dim=1)
        return h[:, -1, :]  # 'last'
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D_orig) - audio features
        Returns:
            h: (B, d_enc) - encoded representation
        """
        # LayerNorm on inputs
        x = self.LN(x)
        
        # Soft permutation (sequence augmentation)
        if self.training and self.use_softperm:
            x = self.sperm(x)
            
        # Align and batch norm
        x = self._align(x)
        if isinstance(self.BN, nn.BatchNorm1d):
            x = self.BN(x.transpose(1, 2)).transpose(1, 2)
            
        # Project and scale
        x = self.proj(x) * self.embed_scale
        
        # Positional embedding + dropout
        x = x + self.embed_positions(x[:, :, 0])
        x = F.dropout(x, p=self.enc_dropout, training=self.training)
        
        # Time-step masking
        if self.training and self.p_mask > 0.0:
            x = self.mask(x)
            
        # Encode
        h = self.enc(x)  # (B, L, d_enc)
        
        # Pool to single vector
        h = self._pool(h)  # (B, d_enc)
        
        return h


class VideoEncoder(nn.Module):
    """Video-only encoder that reads config from tri_enc.vision_encoder."""
    
    def __init__(self, orig_dim: int, seq_len: int, tri_enc_config: Dict):
        super().__init__()
        self.orig_dim = orig_dim
        self.seq_len = seq_len
        
        # Get vision encoder specific configuration
        config = tri_enc_config.get("vision_encoder", {})
        
        # Extract vision encoder configuration with defaults
        self.d_enc = config.get("d_enc", 64)
        self.n_head = config.get("n_head", 4)
        self.nlevels = config.get("nlevels", 3)
        self.maxlen = config.get("maxlen", 50)
        self.enc_dropout = config.get("enc_dropout", 0.1)
        self.use_ln = config.get("use_ln", False)
        self.use_bn = config.get("use_bn", False)
        self.use_softperm = config.get("use_softperm", False)
        self.p_perm = config.get("p_perm", 0.0)
        self.p_mask = config.get("p_mask", 0.0)
        self.pooling = config.get("pooling", "mean")
        
        # Normalization
        self.LN = nn.LayerNorm(orig_dim) if self.use_ln else nn.Identity()
        self.BN = nn.BatchNorm1d(orig_dim) if self.use_bn else nn.Identity()
        
        # Projection and positional encoding
        self.proj = nn.Linear(orig_dim, self.d_enc, bias=False)
        self.embed_scale = math.sqrt(self.d_enc)
        self.embed_positions = SinusoidalPositionalEmbedding(self.d_enc)
        
        # Augmentation
        if self.use_softperm:
            self.sperm = SoftPerm_Fast(p_feat=self.p_perm, maxlen=seq_len)
        self.mask = Masking(self.p_mask)
        
        # Encoder
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
        
    def _align(self, x: torch.Tensor) -> torch.Tensor:
        """Align sequence to maxlen via chunk-mean pooling."""
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
        """Pool sequence to single vector."""
        if self.pooling == "mean":
            return h.mean(dim=1)
        return h[:, -1, :]  # 'last'
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D_orig) - video features
        Returns:
            h: (B, d_enc) - encoded representation
        """
        # LayerNorm on inputs
        x = self.LN(x)
        
        # Soft permutation (sequence augmentation)
        if self.training and self.use_softperm:
            x = self.sperm(x)
            
        # Align and batch norm
        x = self._align(x)
        if isinstance(self.BN, nn.BatchNorm1d):
            x = self.BN(x.transpose(1, 2)).transpose(1, 2)
            
        # Project and scale
        x = self.proj(x) * self.embed_scale
        
        # Positional embedding + dropout
        x = x + self.embed_positions(x[:, :, 0])
        x = F.dropout(x, p=self.enc_dropout, training=self.training)
        
        # Time-step masking
        if self.training and self.p_mask > 0.0:
            x = self.mask(x)
            
        # Encode
        h = self.enc(x)  # (B, L, d_enc)
        
        # Pool to single vector
        h = self._pool(h)  # (B, d_enc)
        
        return h


class BimodalEncoder(nn.Module):
    """Bimodal audio-visual encoder that reads config from tri_enc.bimodal_encoder."""
    
    def __init__(self, orig_d_a: int, orig_d_v: int, l_a: int, l_v: int, tri_enc_config: Dict):
        super().__init__()
        self.orig_d_a = orig_d_a
        self.orig_d_v = orig_d_v
        self.l_a = l_a
        self.l_v = l_v
        
        # Get bimodal encoder specific configuration
        config = tri_enc_config.get("bimodal_encoder", {})
        
        # Extract bimodal encoder configuration with defaults
        self.d_a = self.d_v = config.get("d_enc", 64)
        self.d_enc_out = config.get("d_enc_out", 64)
        self.n_head = config.get("n_head", 4)
        self.nlevels = config.get("nlevels", 3)
        self.maxlen = config.get("maxlen", 50)
        self.enc_dropout = config.get("enc_dropout", 0.1)
        self.use_ln = config.get("use_ln", False)
        self.use_bn = config.get("use_bn", False)
        self.use_softperm = config.get("use_softperm", False)
        self.p_perm = config.get("p_perm", 0.0)
        self.tf_fusion = config.get("tf_fusion", False)
        
        # Masking config
        use_m3 = config.get("use_m3", False)
        if isinstance(use_m3, dict):
            self.p_a = use_m3.get("p_a", 0.0)
            self.p_v = use_m3.get("p_v", 0.0)
            self.use_m3 = True
        else:
            self.p_a = self.p_v = 0.0
            self.use_m3 = False
            
        # Batch norm config
        use_bn_config = config.get("use_bn_config", {})
        if use_bn_config and self.use_bn:
            self.use_bn_a = use_bn_config.get("use_bn_a", False)
            self.use_bn_v = use_bn_config.get("use_bn_v", False)
        elif self.use_bn:
            self.use_bn_a = self.use_bn_v = True
        else:
            self.use_bn_a = self.use_bn_v = False
            
        # Normalization layers
        if self.use_ln:
            self.LN_a = nn.LayerNorm(orig_d_a)
            self.LN_v = nn.LayerNorm(orig_d_v)
            
        if self.use_bn:
            self.BN_a = nn.BatchNorm1d(orig_d_a) if self.use_bn_a else nn.Identity()
            self.BN_v = nn.BatchNorm1d(orig_d_v) if self.use_bn_v else nn.Identity()
            
        # Masking
        if self.use_m3:
            self.mask_a = Masking(self.p_a)
            self.mask_v = Masking(self.p_v)
            
        # Soft permutation
        if self.use_softperm:
            self.a_sperm = SoftPerm_Fast(p_feat=self.p_perm, maxlen=l_a)
            self.v_sperm = SoftPerm_Fast(p_feat=self.p_perm, maxlen=l_v)
            
        # Projections
        self.proj_a = nn.Linear(orig_d_a, self.d_a, bias=False)
        self.proj_v = nn.Linear(orig_d_v, self.d_v, bias=False)
        
        # Positional embeddings
        self.embed_scale_a = math.sqrt(self.d_a)
        self.embed_scale_v = math.sqrt(self.d_v)
        self.embed_positions_a = SinusoidalPositionalEmbedding(self.d_a)
        self.embed_positions_v = SinusoidalPositionalEmbedding(self.d_v)
        
        # Encoders
        self.enc_a = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_a,
                nhead=self.n_head,
                dropout=self.enc_dropout,
                dim_feedforward=4 * self.d_a,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ),
            num_layers=self.nlevels
        )
        
        self.enc_v = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_v,
                nhead=self.n_head,
                dropout=self.enc_dropout,
                dim_feedforward=4 * self.d_v,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ),
            num_layers=self.nlevels
        )
        
        # Fusion
        combined_dim = self.d_a + self.d_v
        if not self.tf_fusion:
            self.fusion = nn.Linear(combined_dim, self.d_enc_out)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(combined_dim, self.d_enc_out),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.d_enc_out,
                        nhead=self.n_head,
                        dropout=self.enc_dropout,
                        dim_feedforward=4 * self.d_enc_out,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True
                    ),
                    num_layers=1
                ),
            )
    
    def align(self, x: torch.Tensor) -> torch.Tensor:
        """Align sequence to maxlen."""
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
        
    def forward(self, x_a: torch.Tensor, x_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_a: (B, L_a, D_a)
            x_v: (B, L_v, D_v)
        Returns:
            h: (B, d_enc_out) - fused representation
        """
        # Layer norm
        if self.use_ln:
            x_a = self.LN_a(x_a)
            x_v = self.LN_v(x_v)
            
        # Soft permutation
        if self.training and self.use_softperm:
            x_a = self.a_sperm(x_a)
            x_v = self.v_sperm(x_v)
            
        # Batch norm and align
        if self.use_bn:
            x_a = self.BN_a(self.align(x_a).transpose(1, 2)).transpose(1, 2)
            x_v = self.BN_v(self.align(x_v).transpose(1, 2)).transpose(1, 2)
        else:
            x_a = self.align(x_a)
            x_v = self.align(x_v)
            
        # Project and scale
        x_a = self.proj_a(x_a) * self.embed_scale_a
        x_v = self.proj_v(x_v) * self.embed_scale_v
        
        # Positional embeddings + dropout
        x_a += self.embed_positions_a(x_a[:, :, 0])
        x_a = F.dropout(x_a, p=self.enc_dropout, training=self.training)
        x_v += self.embed_positions_v(x_v[:, :, 0])
        x_v = F.dropout(x_v, p=self.enc_dropout, training=self.training)
        
        # Modality masking
        if self.use_m3 and self.training:
            if self.p_a > 0:
                x_a = self.mask_a(x_a)
            if self.p_v > 0:
                x_v = self.mask_v(x_v)
                
        # Encode
        x_a = self.enc_a(x_a)
        x_v = self.enc_v(x_v)
        
        # Fusion
        if not self.tf_fusion:
            x_f = torch.cat((x_a, x_v), dim=2)[:, -1, :]  # keep last
            x_f = self.fusion(x_f)
        else:
            x_f = torch.cat((x_a, x_v), dim=2)
            x_f = self.fusion(x_f)[:, 0, :]
            
        return x_f


class TRI_ENC(nn.Module):
    """
    Tri-modal encoder that jointly trains:
    1. Audio-only encoder
    2. Vision-only encoder  
    3. Audio-Visual bimodal encoder
    
    Each encoder produces logits, combined via weighted loss.
    """
    
    def __init__(self, args):
        super().__init__()
        
        # Meta info
        _, self.orig_d_a, self.orig_d_v = args.feature_dims
        _, self.l_a, self.l_v = args.seq_lens
        
        # Config
        cfg = args.get("trienc", {})
        
        # Loss weights
        self.lambda_audio = cfg.get("lambda_audio", 1.0)
        self.lambda_vision = cfg.get("lambda_vision", 1.0) 
        self.lambda_bimodal = cfg.get("lambda_bimodal", 1.0)
        
        # Audio encoder
        self.audio_encoder = AudioEncoder(self.orig_d_a, self.l_a, cfg)
        self.audio_head = nn.Linear(self.audio_encoder.d_enc, 1)
        
        # Video encoder  
        self.video_encoder = VideoEncoder(self.orig_d_v, self.l_v, cfg)
        self.video_head = nn.Linear(self.video_encoder.d_enc, 1)
        
        # Bimodal encoder
        self.bimodal_encoder = BimodalEncoder(
            self.orig_d_a, self.orig_d_v, self.l_a, self.l_v, cfg
        )
        self.bimodal_head = nn.Linear(self.bimodal_encoder.d_enc_out, 1)
        
    def forward(self, _, x_a: torch.Tensor, x_v: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_a: (B, L_a, D_a) - audio features
            x_v: (B, L_v, D_v) - vision features
            
        Returns:
            Dict with keys: 'audio', 'vision', 'bimodal' containing logits
        """
        # Get encoded representations
        h_a = self.audio_encoder(x_a)      # (B, d_enc_a)
        h_v = self.video_encoder(x_v)      # (B, d_enc_v) 
        h_av = self.bimodal_encoder(x_a, x_v)  # (B, d_enc_out)
        
        # Generate logits
        logits_a = self.audio_head(h_a)    # (B, 1)
        logits_v = self.video_head(h_v)    # (B, 1)
        logits_av = self.bimodal_head(h_av)  # (B, 1)
        
        return {
            'audio': logits_a,
            'vision': logits_v, 
            'bimodal': logits_av
        }