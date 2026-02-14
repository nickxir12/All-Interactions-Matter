import math
import random
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F
import os

from enum import Enum
from typing import Optional, Dict, Union
from dataclasses import dataclass

import matplotlib.pyplot as plt
#import seaborn as sns

import torch.distributions as D
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from MMSA.models.singleTask.MMGPT import (AV_Enc,
                                          MultiheadCrossAttention,
                                          LoRA_MLP, LayerNorm, MLP)

###################################################################################################
# Helper Code
###################################################################################################
class GaussianNoise(nn.Module):
    def __init__(self, stddev, mean=0.0):
        """
        Additive Gaussian Noise layer
        Args:
            stddev (float): the standard deviation of the distribution
            mean (float): the mean of the distribution
        """
        super().__init__()
        self.stddev = stddev
        self.mean = mean

    def forward(self, x):
        if self.training:
            noise = torch.normal(mean=self.mean, std=self.stddev, size=x.size(), device=x.device)
            return x + noise
        return x

    def __repr__(self):
        return '{} (mean={}, stddev={})'.format(self.__class__.__name__,
                                                str(self.mean),
                                                str(self.stddev))

def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work

def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "smolLM": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
}

def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1 :])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


###################################################################################################
# Core Layers
###################################################################################################
class SoftPerm(nn.Module):
    def __init__(
        self,
        p_apply: float = 0.5,
        p_feat: float = 0.2,
        maxlen: Optional[int] = 50,
        n_bn: Optional[int] = 16,
    ):
        """
        SoftPerm_Fast: SoftPermutation (Fast), randomly samples a Bernoulli distribution
        with hyperparam $p$. By sampling, it gets the `copy_area`
        (float), which denotes the proportion (=area) of feature dimensions
        that are going to be resampled (i.e how many features are going to be
        permuted).

        Args:
            p_t_mod (List[float]): Resample probabilities for each modality across timesteps.
                default=1's which means that we resample at every timestep
            alpha (List[float]): used in sampling uniform distribution for each modality.
                Encodes the amount of features to be resampled. In order to resample the same
                distribution one should give a list of multiple variables as input in [0,1).
            maxlen (int): must be given as input and refers to the maximum sequence length
                In this implementation we also assume that the sequences are already zero-padded
        """
        super().__init__()
        self.p_apply = p_apply
        self.p_feat = p_feat  # resampled features
        self.maxlen = maxlen
        self.n_bn = n_bn
        
        # weights
        self.w = torch.tensor([1/maxlen]*maxlen, device="cuda")
        # probs --> '1', non-resampled features
        self.bern = D.bernoulli.Bernoulli(probs=1 - torch.tensor(self.p_feat, device="cuda"))

    def forward(self, x):
        
        if self.training and self.p_feat > 0 and self.p_apply > random.random():
            B, seqlen, D = x.size()

            # Only permute the text span (exclude BN tokens)
            lang_len = seqlen - self.n_bn
            if lang_len <= 1:
                return x

            # Build probs matching the CURRENT lang_len (not maxlen)
            # (keep it uniform; or slice self.w if you really want that weighting)
            probs = torch.full((lang_len,), 1.0 / lang_len, device=x.device)

            # Sample a permutation over [0..lang_len-1]
            perm = torch.multinomial(probs, lang_len, replacement=False)

            x_lang = x[:, :lang_len, :]
            x_perm = x_lang[:, perm, :]

            # Feature-wise Bernoulli mask on the same device as x
            area_mask = torch.bernoulli(
                torch.full((B, D), 1 - self.p_feat, device=x.device)
            ).unsqueeze(1)  # (B,1,D)

            x[:, :lang_len, :] = area_mask * x_lang + (1 - area_mask) * x_perm

        return x


class _LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class _LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = config.get("n_up", 4 * self.hidden_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # if self.config.pretraining_tp > 1:
        #     slice = self.intermediate_size // self.config.pretraining_tp
        #     gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        #     up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        #     down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        #     gate_proj = torch.cat(
        #         [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        #     )
        #     up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        #     intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        #     down_proj = [
        #         F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        #     ]
        #     down_proj = sum(down_proj)
        # else:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
    
class LoRALinear(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Actual trainable parameters
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(in_features, r))
            self.lora_B = nn.Parameter(torch.zeros(r, out_features))
            self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        # x: (B, L, D_in) --> (B, L, D_out)
        x = self.lora_dropout(x)
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = x * self.scaling
        return x


class _LlamaMLP_LoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_embd
        self.intermediate_size = config.get("n_up", 4 * self.hidden_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

        #Lora params
        self.lora_gate_proj = LoRALinear(
            self.hidden_size,
            self.intermediate_size,
            self.config.lora.r,
            self.config.lora.lora_alpha,
            self.config.lora.lora_dropout,
        )
        self.lora_up_proj = LoRALinear(
            self.hidden_size,
            self.intermediate_size,
            self.config.lora.r,
            self.config.lora.lora_alpha,
            self.config.lora.lora_dropout,
        )
        self.lora_down_proj = LoRALinear(
            self.intermediate_size,
            self.hidden_size,
            self.config.lora.r,
            self.config.lora.lora_alpha,
            self.config.lora.lora_dropout,
        )

    def forward(self, x):
        x_gate = self.act_fn(self.gate_proj(x) + self.lora_gate_proj(x))
        x_up = self.up_proj(x) + self.lora_up_proj(x)
        x_down = x_gate + x_up
        x_down = self.down_proj(x_down) + self.lora_down_proj(x_down)
        return x_down



class MMBlock_discrete(nn.Module):
    """extension of the Block class to multimodal gated cross attention
    """
    def __init__(self, config, idx=-1):
        super().__init__()
        self.idx = idx
        self.lm_flavor = config.get("type", "llama")

        ## OUR CHANGES
        self.accumilation = config.get(
            "accumilation"
        )  # Added tokens for AV as accumilator ablation, if > 0 then we have that many accumilation tokens, replacing the n_bn_fusion tokens

        ##END

        if "gpt" in self.lm_flavor:
            self.kdim_a = config.get("kv_dim_a", 16)
            self.kdim_v = config.get("kv_dim_v", 16)
            self.kdim_av = config.get("kv_dim_av", 30)
            self.ln_1 = LayerNorm(config.n_embd, config.bias)
            self.ln_2 = LayerNorm(config.n_embd, config.bias)
            
            #self.attn = MultiheadCrossAttention(
            #    config.n_head,
            #   config.n_embd,  # the same as query size
            #    self.kdim,  # multimodal dimension, k, v
            #    config.dropout,
            #)

            ## OUR CHANGES
            self.audio_attn = MultiheadCrossAttention(
                config.n_head, 
                config.n_embd, 
                self.kdim_a, 
                config.dropout
            )
            self.video_attn = MultiheadCrossAttention(
                config.n_head, 
                config.n_embd, 
                self.kdim_v, 
                config.dropout
            )

            self.av_attn = MultiheadCrossAttention(
                config.n_head, 
                config.n_embd, 
                self.kdim_av, 
                config.dropout
            )
            ## END 

            if config.use_lora:
                self.mlp = LoRA_MLP(config)
            else:
                self.mlp = MLP(config)
        else:  ##MOSI related
            self.ln_1 = _LlamaRMSNorm(config.n_embd)
            # self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.kdim = config.get("kv_dim", config.n_embd)
            self.attn = MultiheadCrossAttention(
                config.n_head,
                config.n_embd,  # the same as query size
                self.kdim,  # multimodal dimension, k, v
                config.dropout,
            )
            self.ln_2 = _LlamaRMSNorm(config.n_embd)
            # self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            # if config.get("use_lora", False):
            #     self.mlp = LoRA_MLP(config)
            # else:
            self.mlp = _LlamaMLP_LoRA(config) # adopt this to initialize at the same
        
        ###########################################################################################
        # common parameters
        print(f"Ongoing with ----- {config.gating} ----- gating")
        self.gate_1 = nn.Sigmoid()
        self.gate_2 = nn.Sigmoid()
        ## OUR CHANGES
        self.gate_3 = nn.Sigmoid()
        ## END

        # self.gate_1 = nn.Tanh()
        # self.gate_2 = nn.Tanh()
        # # init gate at ~0.5

        #init_value = config.get("init_gate", 0)
        #print(f"idx is {idx}")
        #print(f"{init_value}")
        #if config.get("init_gate_2", None):
        #    init_value_2 = config.get("init_gate_2")
        #else:
        #    init_value_2 = init_value
        #if self.idx == -1:
        #    self.alpha_1 = nn.Parameter(init_value * torch.ones(1))
        #    # self.alpha_1 = nn.Parameter(torch.zeros(1))
        #    self.alpha_2 = nn.Parameter(init_value_2 * torch.ones(1))
            # self.alpha_2 = nn.Parameter(torch.zeros(1))
        #else:
        #    self.alpha_1 = nn.Parameter(init_value[idx] * torch.ones(1))
        #    self.alpha_2 = nn.Parameter(init_value_2[idx] * torch.ones(1))

        ##  OUR CHANGES
        init_value = config.get("init_gate", 0)
        init_value_2 = config.get("init_gate_2", init_value)
        init_value_3 = config.get("init_gate_3", init_value)
        init_value_4 = config.get("init_gate_4", init_value)

        if isinstance(init_value, list):
            self.alpha_1 = nn.Parameter(torch.tensor([init_value[self.idx]]))
        else:
            self.alpha_1 = nn.Parameter(torch.tensor([init_value]))

        if isinstance(init_value_2, list):
            self.alpha_2 = nn.Parameter(torch.tensor([init_value_2[self.idx]]))
        else:
            self.alpha_2 = nn.Parameter(torch.tensor([init_value_2]))

        if isinstance(init_value_3, list):
            self.alpha_3 = nn.Parameter(torch.tensor([init_value_3[self.idx]]))
        else:
            self.alpha_3 = nn.Parameter(torch.tensor([init_value_3]))

        if isinstance(init_value_4, list):
            self.alpha_4 = nn.Parameter(torch.tensor([init_value_4[self.idx]]))
        else:
            self.alpha_4 = nn.Parameter(torch.tensor([init_value_4]))
        ## END
        # combined cross-attention
        self.combine = config.get("combine", False)
        
    def forward(
        self, x_q_audio, x_q_video, x_q_av, x_audio, x_video, x_av, x_prev=None
    ):
        """
        Args:
            x_q_audio: audio query tokens (fusion tokens)
            x_q_video: video query tokens (fusion tokens)
            x_q_av: audio - visual query tokens (fusion tokens)
            x_audio: audio keys & values (encoder) | None if it is missing
            x_video: video keys & values (encoder) | None if it is missing
            x_av: audio - visual keys & values (encoder) | None if it is missing
            x_prev: previous hidden state (optional, for combining)
        """
        x_q_audio = x_q_audio[0] if x_audio is not None else None  # Unpack query tensor
        x_q_video = x_q_video[0] if x_video is not None else None  # Unpack query tensor
        x_q_av = x_q_av[0] if x_av is not None or self.accumilation > 0 else None
        norm_x_q_audio = self.ln_1(x_q_audio) if x_q_audio is not None else None
        norm_x_q_video = self.ln_1(x_q_video) if x_q_video is not None else None
        norm_x_q_av = self.ln_1(x_q_av) if x_q_av is not None else None


        # Compute distinct audio, video and av cross-attentions
        audio_ca = (
            self.audio_attn(norm_x_q_audio, x_audio) if x_audio is not None else None
        )
        video_ca = (
            self.video_attn(norm_x_q_video, x_video) if x_video is not None else None
        )
        av_ca = self.av_attn(norm_x_q_av, x_av) if x_av is not None else None

        x_a_ = (
            x_q_audio + self.gate_1(self.alpha_1) * audio_ca
            if audio_ca is not None
            else None
        )
        x_v_ = (
            x_q_video + self.gate_2(self.alpha_2) * video_ca
            if video_ca is not None
            else None
        )
        if av_ca is not None:
            x_av_ = x_q_av + self.gate_3(self.alpha_3) * av_ca
        elif self.accumilation > 0:
            x_av_ = norm_x_q_av  # pass with risidual connection, untouched from cross attention
        else:
            x_av_ = None

        # all cases of ablation
        if (x_a_ is not None) and (x_v_ is not None) and (x_av_ is not None):
            x = torch.cat((x_a_, x_v_, x_av_), dim=1)
        elif (x_a_ is not None) and (x_v_ is not None) and (x_av_ is None):
            x = torch.cat((x_a_, x_v_), dim=1)
        elif (x_a_ is not None) and (x_v_ is None) and (x_av_ is not None):
            x = torch.cat((x_a_, x_av_), dim=1)
        elif (x_a_ is None) and (x_v_ is not None) and (x_av_ is not None):
            x = torch.cat((x_v_, x_av_), dim=1)
        else:
            x = None

        # Optional combination with previous states
        fused_parts = []
        if x_a_ is not None and x_a_.size(1) > 0:
            fused_parts.append(x_a_)
        if x_v_ is not None and x_v_.size(1) > 0:
            fused_parts.append(x_v_)
        if x_av_ is not None and x_av_.size(1) > 0:
            fused_parts.append(x_av_)

        if self.combine and x_prev is not None:
            if len(fused_parts) == 0:
                # nothing new to fuse → pure passthrough
                return (x_prev,)
            x_comb = torch.cat([x_prev] + fused_parts, dim=1)
            x = x_comb + self.gate_2(self.alpha_4) * self.mlp(self.ln_2(x_comb))
        else:
            if len(fused_parts) == 0:
                # tell caller that no BN produced in bottleneck-only path
                return (None,)
            x_cat = torch.cat(fused_parts, dim=1)
            x = x_cat + self.gate_2(self.alpha_4) * self.mlp(self.ln_2(x_cat))

        return (x,)


class msaLMLayer(nn.Module):
    """
    msaLM layer is a wrapper around the MMBlock and native decoder LMLayer.
    """

    def __init__(
        self, 
        ca_layer, 
        decoder_layer,
        nv_bn_fusion=-1,
        na_bn_fusion=-1,
        n_bn_fusion=-1,
        max_len=50,
        combine=False, 
        lm_flavor="llama", 
        ldrop=0.0,
        accumilation=0,
    ):
        super().__init__()
        self.ca_layer = ca_layer
        self.decoder_layer = decoder_layer
        self.n_bn_fusion = n_bn_fusion
        self.nv_bn_fusion = nv_bn_fusion
        self.na_bn_fusion = na_bn_fusion
        self.max_len = max_len
        self.combine = combine
        self.av_x = None
        self.a_x = None
        self.v_x = None
        self.lm_flavor = lm_flavor
        self.accumilation = accumilation
        # self.ldrop = ldrop
        self.tie_ffw()
    
    def tie_ffw(self):
        if self.ca_layer is not None:
            print("COpying---------------------------------")
            print("LM FLAVOR: ", type(self.lm_flavor), self.lm_flavor)
            if "gpt" in self.lm_flavor:
                # print(self.decoder_layer)
                assert self.ca_layer.mlp.c_fc.weight.shape == self.decoder_layer.mlp.c_fc.weight.shape[::-1]
                with torch.no_grad():
                    self.ca_layer.mlp.c_fc.weight.copy_(
                        self.decoder_layer.mlp.c_fc.weight.data.t()
                    )
                assert self.ca_layer.mlp.c_proj.weight.shape == self.decoder_layer.mlp.c_proj.weight.shape[::-1]
                with torch.no_grad():
                    self.ca_layer.mlp.c_proj.weight.copy_(
                        self.decoder_layer.mlp.c_proj.weight.data.t()
                    )
            else:
                # llama flavor
                # Assuming all the weight attributes are PyTorch tensors or parameters
                self.ca_layer.mlp.gate_proj.weight.data.copy_(
                    self.decoder_layer.mlp.gate_proj.weight.data
                )

                self.ca_layer.mlp.up_proj.weight.data.copy_(
                    self.decoder_layer.mlp.up_proj.weight.data
                )

                self.ca_layer.mlp.down_proj.weight.data.copy_(
                    self.decoder_layer.mlp.down_proj.weight.data
                )
            
    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return (
            (self.av_x is not None) or (self.a_x is not None) or (self.v_x is not None)
        )

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_audio(self, a_x):
        self.a_x = a_x

    def condition_video(self, v_x):
        self.v_x = v_x

    def condition_av(self, av_x):
        self.av_x = av_x

    # #ORIGINAL FORWARD

    @staticmethod
    def _slice_span(x, start, length):
        """Safe slice: returns empty [B,0,D] if length<=0."""
        B, _, D = x.shape
        if length <= 0:
            return x.new_zeros((B, 0, D))
        return x[:, start:start+length, :]
    
    def forward(self, lang_x, attention_mask=None, **decoder_layer_kwargs):
        # 1) native decoder
        native_out = self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        hidden_states = native_out[0] if isinstance(native_out, (tuple, list)) else native_out

        # 2) CA path (subset-safe)
        if self.ca_layer is not None:
            x = hidden_states
            lang_part = self._slice_span(x, 0, self.max_len)

            start_a = self.max_len
            audio_tokens = self._slice_span(x, start_a, max(0, self.na_bn_fusion))

            start_v = start_a + audio_tokens.size(1)
            video_tokens = self._slice_span(x, start_v, max(0, self.nv_bn_fusion))

            start_av = start_v + video_tokens.size(1)
            av_len = max(0, self.accumilation if self.accumilation > 0 else self.n_bn_fusion)
            av_tokens = self._slice_span(x, start_av, av_len)

            if self.combine:
                out = self.ca_layer(
                    (audio_tokens,), (video_tokens,), (av_tokens,),
                    self.a_x, self.v_x, self.av_x,
                    lang_part,   # x_prev
                )
                fused = out[0] if isinstance(out, (tuple, list)) else out
                # fallback if CA produced nothing (e.g., all encodings None)
                updated_lang_x = fused if (fused is not None) else hidden_states
            else:
                out = self.ca_layer(
                    (audio_tokens,), (video_tokens,), (av_tokens,),
                    self.a_x, self.v_x, self.av_x
                )
                bn_x = out[0] if isinstance(out, (tuple, list)) else out
                # if bn_x is None or empty, keep native states
                if (bn_x is None) or (bn_x.size(1) == 0):
                    updated_lang_x = hidden_states
                else:
                    updated_lang_x = torch.cat((lang_part, bn_x), dim=1)

            hidden_states = updated_lang_x

        # 3) preserve HF return contract
        if isinstance(native_out, (tuple, list)):
            present = native_out[1] if len(native_out) > 1 else None
            attn    = native_out[2] if len(native_out) > 2 else None
            if attn is not None:
                return (hidden_states, present, attn)
            elif present is not None:
                return (hidden_states, present)
            else:
                return (hidden_states,)
        else:
            return hidden_states
            

            


    # FORWARD THAT WORKS ONLY WITH MOSI

    # def forward(
    #     self,
    #     lang_x,
    #     attention_mask=None,
    #     **decoder_layer_kwargs,
    # ):
    #     # 1) Run the original decoder layer
    #     lang_x = self.decoder_layer(
    #         lang_x, attention_mask=attention_mask, **decoder_layer_kwargs
    #     )
    #     # 2) UNWRAP to Tensor if needed
    #     if isinstance(lang_x, (tuple, list)):
    #         lang_x = lang_x[0]
    #     elif hasattr(lang_x, "hidden_states"):
    #         lang_x = lang_x.hidden_states
    #     elif hasattr(lang_x, "last_hidden_state"):
    #         lang_x = lang_x.last_hidden_state

    #     # 3) Cross attention
    #     if self.ca_layer is not None:
    #         if self.av_x is None:
    #             raise ValueError("av_x must be conditioned before forward pass")

    #         if self.n_bn_fusion > 0:
    #             # split language vs BN tokens
    #             lang_part = lang_x[:, :self.max_len, :]
    #             bn_part   = lang_x[:, self.max_len:, :]

    #             if self.combine:
    #                 out = self.ca_layer((bn_part,), self.av_x, lang_part)
    #                 bn_updated = out[0] if isinstance(out, (tuple, list)) else out
    #                 updated_lang_x = bn_updated
    #             else:
    #                 bn_x = self.ca_layer((bn_part,), self.av_x)
    #                 bn_x = bn_x[0] if isinstance(bn_x, (tuple, list)) else bn_x
    #                 updated_lang_x = torch.cat((lang_part, bn_x), dim=1)

    #             lang_x = updated_lang_x                 # <-- DO NOT wrap into a tuple
    #         else:
    #             out = self.ca_layer(lang_x, self.av_x)
    #             lang_x = out[0] if isinstance(out, (tuple, list)) else out

    #     # 4) Final sanity
    #     assert torch.is_tensor(lang_x), f"decoder wrapper must return Tensor, got {type(lang_x)}"
    #     return lang_x


    #FORWARD THAT WORKS WITH BOTH MOSI AND SIMS

    # def forward(self, *call_args, **call_kwargs):
    #     """
    #     Decoder-layer wrapper forward that works for GPT-2 and LLaMA.
    #     - Accepts positional/keyword inputs.
    #     - Forwards only args the inner decoder supports (via inspect).
    #     - Applies your cross-attention (with optional BN-fusion).
    #     - RETURNS:
    #         * GPT-2: (hidden_states, present) or (hidden_states, present, attn)
    #         * LLaMA: hidden_states  (to match your previous MOSI behavior)
    #     """

    #     # ---- (A) Read inputs from positional/kwargs ----
    #     hidden_states = call_args[0] if len(call_args) > 0 else call_kwargs.get("hidden_states")

    #     # Common
    #     layer_past = call_kwargs.get("layer_past", call_kwargs.get("past_key_value"))
    #     attention_mask = call_kwargs.get("attention_mask")
    #     head_mask = call_kwargs.get("head_mask")
    #     encoder_hidden_states = call_kwargs.get("encoder_hidden_states")
    #     encoder_attention_mask = call_kwargs.get("encoder_attention_mask")
    #     use_cache = call_kwargs.get("use_cache", False)
    #     output_attentions = call_kwargs.get("output_attentions", False)

    #     # LLaMA-specific (HF versions may pass these)
    #     position_ids = call_kwargs.get("position_ids")
    #     position_embeddings = call_kwargs.get("position_embeddings")  # expected as (cos, sin)
    #     cache_position = call_kwargs.get("cache_position")

    #     # ---- (B) Build kwargs for the wrapped decoder layer ONLY from its signature ----
    #     sig = inspect.signature(self.decoder_layer.forward)
    #     params = sig.parameters
    #     dl_kwargs = {}
    #     if "attention_mask" in params: dl_kwargs["attention_mask"] = attention_mask
    #     if "layer_past" in params: dl_kwargs["layer_past"] = layer_past
    #     elif "past_key_value" in params: dl_kwargs["past_key_value"] = layer_past
    #     if "head_mask" in params: dl_kwargs["head_mask"] = head_mask
    #     if "encoder_hidden_states" in params: dl_kwargs["encoder_hidden_states"] = encoder_hidden_states
    #     if "encoder_attention_mask" in params: dl_kwargs["encoder_attention_mask"] = encoder_attention_mask
    #     if "use_cache" in params: dl_kwargs["use_cache"] = use_cache
    #     if "output_attentions" in params: dl_kwargs["output_attentions"] = output_attentions
    #     if "position_ids" in params and (position_ids is not None): dl_kwargs["position_ids"] = position_ids
    #     if "position_embeddings" in params and (position_embeddings is not None): dl_kwargs["position_embeddings"] = position_embeddings
    #     if "cache_position" in params and (cache_position is not None): dl_kwargs["cache_position"] = cache_position

    #     # ---- (C) Call the original HF decoder layer (NO *args to avoid double-binding) ----
    #     base_out = self.decoder_layer(hidden_states, **dl_kwargs)

    #     # ---- (D) Unwrap HF-style return ----
    #     if isinstance(base_out, (tuple, list)):
    #         hs = base_out[0]
    #         present = base_out[1] if len(base_out) > 1 else None
    #         attn = base_out[2] if (output_attentions and len(base_out) > 2) else None
    #     else:
    #         # Defensive attributes some models use
    #         if hasattr(base_out, "hidden_states"):
    #             hs = base_out.hidden_states
    #         elif hasattr(base_out, "last_hidden_state"):
    #             hs = base_out.last_hidden_state
    #         else:
    #             hs = base_out
    #         present, attn = None, None

    #     # ---- (E) Your cross-attention (+ BN fusion) ----
    #     if self.ca_layer is not None:
    #         if self.av_x is None:
    #             raise ValueError("av_x must be conditioned before forward pass (call condition_av_x).")

    #         if self.n_bn_fusion > 0:
    #             if not isinstance(self.max_len, int) or self.max_len <= 0:
    #                 raise ValueError("max_len must be > 0 when using n_bn_fusion.")

    #             lang_part = hs[:, :self.max_len, :]
    #             bn_part   = hs[:, self.max_len:, :]

    #             if self.combine:
    #                 out = self.ca_layer((bn_part,), self.av_x, lang_part)
    #                 bn_updated = out[0] if isinstance(out, (tuple, list)) else out
    #                 hs = bn_updated
    #             else:
    #                 bn_x = self.ca_layer((bn_part,), self.av_x)
    #                 bn_x = bn_x[0] if isinstance(bn_x, (tuple, list)) else bn_x
    #                 hs = torch.cat((lang_part, bn_x), dim=1)
    #         else:
    #             out = self.ca_layer(hs, self.av_x)
    #             hs = out[0] if isinstance(out, (tuple, list)) else out

    #         if str(getattr(self, "lm_flavor", "llama")).startswith("gpt"):
    #             return (hs, present, attn) if output_attentions else (hs, present)
    #         else:
    #             return hs



class BNEmbedding(nn.Module):
    def __init__(
        self, na_bn_fusion=-1, nv_bn_fusion=-1, n_bn_fusion=-1, d=1024, accumilation=0
    ):
        super().__init__()
        self.na_bn_fusion = na_bn_fusion
        self.nv_bn_fusion = nv_bn_fusion
        self.n_bn_fusion = n_bn_fusion
        self.accumilation = accumilation
        print("#####################################################################")
        print(
            type(na_bn_fusion),
            type(nv_bn_fusion),
            type(n_bn_fusion),
            type(accumilation),
            type(d),
        )

        self.bn_embedding_audio = (
            nn.Parameter(torch.zeros(na_bn_fusion, d))
            if self.na_bn_fusion > 0
            else None
        )
        self.bn_embedding_video = (
            nn.Parameter(torch.zeros(nv_bn_fusion, d))
            if self.nv_bn_fusion > 0
            else None
        )
        # either n_bn_fusion > 0 or accumilation > 0, not both. also we can have both 0 to missing AV
        if self.n_bn_fusion > 0:
            self.bn_embedding_av = nn.Parameter(torch.zeros(n_bn_fusion, d))
        elif self.accumilation > 0:
            self.bn_embedding_av = nn.Parameter(torch.zeros(accumilation, d))
        else:
            self.bn_embedding_av = None

        if self.bn_embedding_audio is not None:
            torch.nn.init.normal_(self.bn_embedding_audio, mean=0.0, std=0.02)

        if self.bn_embedding_video is not None:
            torch.nn.init.normal_(self.bn_embedding_video, mean=0.0, std=0.02)

        if self.bn_embedding_av is not None:
            torch.nn.init.normal_(self.bn_embedding_av, mean=0.0, std=0.02)

    ################################################################################################################
    def forward(self, lang_x):
        B = lang_x.size(0)
        tokens = [lang_x]

        if self.bn_embedding_audio is not None:
            audio_tokens = self.bn_embedding_audio.unsqueeze(0).repeat(
                B, 1, 1
            )  # (B, na_bn_fusion, D)
            tokens.append(audio_tokens)

        if self.bn_embedding_video is not None:
            video_tokens = self.bn_embedding_video.unsqueeze(0).repeat(
                B, 1, 1
            )  # (B, nv_bn_fusion, D)
            tokens.append(video_tokens)

        if self.bn_embedding_av is not None:
            av_tokens = self.bn_embedding_av.unsqueeze(0).repeat(
                B, 1, 1
            )  # (B, nv_bn_fusion, D)
            tokens.append(av_tokens)

        return torch.cat(tokens, dim=1)  # (B, L + na + nv + nav, D)


class msaLMEmbedding(nn.Module):
    """
    msaLM layer is a wrapper around the MMBlock and native decoder LMLayer.
    """

    def __init__(
        self,
        embedding,
        seqaug=None,
        na_bn_fusion=-1,
        nv_bn_fusion=-1,
        n_bn_fusion=-1,
        d_embd=1024,
        lm_flavor="llama",
        max_len=50,
        accumilation=0,
    ):
        super().__init__()
        self.embedding = embedding
        # self.positional = positional
        self.seqaug = seqaug
        self.na_bn_fusion = na_bn_fusion
        self.nv_bn_fusion = nv_bn_fusion
        self.n_bn_fusion = n_bn_fusion
        self.accumilation = accumilation
        self.lm_flavor = lm_flavor
        self.max_len = max_len
        self.bn_embedding = BNEmbedding(
            na_bn_fusion, nv_bn_fusion, n_bn_fusion, d_embd, accumilation
        )

    def forward(
        self,
        lang_x,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        # Normal embedding layer
        lang_x = self.embedding(lang_x, **decoder_layer_kwargs)

        # print(f"[msaLMEmbedding] shape after embedding: {lang_x.shape}")

        # sqaug layer
        # 2) sequence augmentation ONLY on the text span
        if self.seqaug is not None and self.training:
            if "gpt" in self.lm_flavor:
                # split: [text | BN-dummy ids]
                text_part = lang_x[:, : self.max_len, :]          # (B, max_len, D)
                bn_part   = lang_x[:, self.max_len:, :]           # (B, extra_bn, D)  <-- untouched
                text_part = self.seqaug(text_part)                # apply SoftPerm here
                lang_x    = torch.cat([text_part, bn_part], dim=1)
            else:
                # LLaMA path has no BN ids appended here, safe to augment whole sequence
                lang_x = self.seqaug(lang_x)

        # bn layer
        if "gpt" in self.lm_flavor:
            true_lang_x = lang_x[:, : self.max_len, :]
            _lang_x = self.bn_embedding(true_lang_x)
        else:
            _lang_x = self.bn_embedding(lang_x)
        lang_x = _lang_x

        return lang_x


class msaLMPositional(nn.Module):
    """
    msaLM layer is a wrapper around the MMBlock and native positional encoding LMLayer.
    """

    def __init__(
        self, positional, n_fusion=8
    ):
        super().__init__()
        self.positional = positional
        self.n_fusion = n_fusion

    def forward(
        self,
        pos_id,
        attention_mask=None,
        **decoder_layer_kwargs,
    ):
        
        # positional embeddings - gpt case
        _, t = pos_id.size()
        device = pos_id.device
        pos = torch.arange(
            0, t, dtype=torch.long, device=device
        )
        pos_emb = self.positional(pos)
        return pos_emb


class msaLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name="model.layers"):
        self.decoder_layers_attr_name = decoder_layers_attr_name
    
    def set_embedding_attr_name(self, embedding_attr_name="model.embed_tokens"):
        self.embedding_attr_name = embedding_attr_name

    def set_positional_attr_name(self, positional_attr_name="transformer.wpe"):
        self.positional_attr_name = positional_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _get_embedding_layers(self):
        return getattr_recursive(self, self.embedding_attr_name)

    def _get_positional_layers(self):
        return getattr_recursive(self, self.positional_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)
    
    def _set_embedding_layers(self, value):
        setattr_recursive(self, self.embedding_attr_name, value)

    def _set_positional_layers(self, value):
        setattr_recursive(self, self.positional_attr_name, value)

    def init_msalm(
        self,
        ca_list,
        config
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """
        self.msa_config = config
        self.set_embedding_attr_name(
            self.msa_config.get("embedding_attr_name", "model.embed_tokens")
        )
        self.set_decoder_layers_attr_name(
            self.msa_config.get("decoder_layers_attr_name", "model.layers")
        )
        self.lm_flavor = self.msa_config["mmgpt"].get("type", "llama")
        if "gpt" in self.lm_flavor:
            self.set_positional_attr_name()
        print(f"ca list is: {ca_list}")
        self.init_embedding_layers()
        if "gpt" in self.lm_flavor:
            self.init_positional_layers()
        self.init_msalm_layers(ca_list)
        self.initialized_msalm = True
        self._use_cached_av_x = False
        ## CHANGES
        self.na_bn_fusion = self.msa_config.get("na_bn_fusion", 0)
        self.nv_bn_fusion = self.msa_config.get("nv_bn_fusion", 0)
        self.n_bn_fusion = self.msa_config.get("n_bn_fusion", 0)
        self.accumilation = self.msa_config.get("accumilation", 0)
        print(
            f"Fusion tokens used are: {self.na_bn_fusion} for Audio, {self.nv_bn_fusion} for Video, and {self.n_bn_fusion} for AV"
        )
        if self.accumilation > 0:
            print(f"{self.accumilation} tokens are used as accumilator")

    def init_msalm_layers(self, ca_list):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        old_decoder_blocks = self._get_decoder_layers()

        ca_layers = nn.ModuleList(
            [
                MMBlock_discrete(self.msa_config["mmgpt"], layer_idx)
                for layer_idx in range(len(ca_list))
            ]
        )

        all_layers = []
        for idx, orig_block in enumerate(old_decoder_blocks):
            # print(f"Parsing decoder block: {idx}")
            # pick up cross-attn if this layer is in ca_list
            if idx in ca_list:
                ca_idx = ca_list.index(idx)
                ca_layer = ca_layers[ca_idx]
            else:
                ca_layer = None

            # instantiate the msaLMLayer with the **original** HF block
            msa_layer = msaLMLayer(
                ca_layer,
                orig_block,
                nv_bn_fusion=self.msa_config["nv_bn_fusion"],
                na_bn_fusion=self.msa_config["na_bn_fusion"],
                n_bn_fusion=self.msa_config["n_bn_fusion"],
                max_len=self.msa_config["max_token_len"],
                combine=self.msa_config["mmgpt"].get("combine", False),
                lm_flavor=self.msa_config["mmgpt"].get("type", "llama"),
                accumilation=self.msa_config.get("accumilation", 0),
            )

            pattern_from_config = self.msa_config["mmgpt"].get(
                "attention_pattern", "original"
            )

            # wrap its decoder_layer for custom masking
            wrapped = FlexibleCSAWrappedBlock(
                msa_layer.decoder_layer,
                self.msa_config["max_token_len"],
                self.msa_config["na_bn_fusion"],
                self.msa_config["nv_bn_fusion"],
                self.msa_config["n_bn_fusion"],
                layer_idx=idx,
                attention_pattern=pattern_from_config,
            )

            # if you want to see it right now:
            if idx == 0:
                print("Pattern for the first layer:", pattern_from_config)
                wrapped.visualize_mask(pattern_from_config)

            # replace the plain decoder_layer with your wrapped one
            msa_layer.decoder_layer = wrapped

            all_layers.append(msa_layer)

        # finally overwrite the model’s decoder layers
        self._set_decoder_layers(nn.ModuleList(all_layers))
    
    def init_embedding_layers(self):
        """
        Re initializes the FlamingoLayers.
        Propagates any changes made to self.gated_corss_attn_layers or self.old_decoder_blocks
        """
        # get pretrained embedding block
        embedding_block = self._get_embedding_layers()
        positional_block = None
        if "gpt" in self.lm_flavor:
            positional_block = self._get_positional_layers()
        if self.msa_config.get("use_seqaug", False):
            print("initializing SoftPerm")
            seqaug_layer = SoftPerm(
                p_apply=self.msa_config["mmgpt"]["p_apply"],
                p_feat=self.msa_config["mmgpt"]["p_perm"],
                n_bn=self.msa_config["n_bn_fusion"],
                maxlen=self.msa_config["max_token_len"]
            )
            # #### ABLATION
            # if self.msa_config["mmgpt"]["ablation"] == "dropout":
            #     # Dropout
            #     print("initializing Embedding Dropout")
            #     seqaug_layer = nn.Dropout(p=self.msa_config["mmgpt"]["p_ablation"])
            # elif self.msa_config["mmgpt"]["ablation"] == "noise":
            #     # White Noise
            #     print("initializing GaussianNoise")
            #     seqaug_layer = GaussianNoise(stddev=self.msa_config["mmgpt"]["p_ablation"])
            # else:
            #     print('No ablation ------------ SHOULD NOT BE HERE --------------')

        else:
            seqaug_layer = None
        
        new_embedding = msaLMEmbedding(
            embedding_block,
            # positional_block,
            seqaug_layer,
            self.msa_config.get("na_bn_fusion", -1), #audio
            self.msa_config.get("nv_bn_fusion", -1), #video
            self.msa_config.get("n_bn_fusion", -1),
            self.msa_config["mmgpt"]["n_embd"],
            self.msa_config["mmgpt"].get("type", 'llama'),
            self.msa_config["max_token_len"],
            self.msa_config["accumilation"],
        )
        self._set_embedding_layers(nn.Sequential(new_embedding))
    
    def init_positional_layers(self):
        # get pretrained embedding block
        positional_block = self._get_positional_layers()
        
        new_positional = msaLMPositional(
            positional_block
        )
        self._set_positional_layers(nn.Sequential(new_positional))
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_msalm:
            raise ValueError(
                "Flamingo layers are not initialized. Please call `init_flamingo` first."
            )

        # package arguments for the other parent's forward. make them all kwargs
        if 'gpt' in self.lm_flavor:
            # gpt2
            b, _ = input_ids.size()
            zero_ids = torch.zeros(
                (b, self.na_bn_fusion + self.nv_bn_fusion + self.n_bn_fusion),
                dtype=torch.long,
                device=input_ids.device
            )
            input_ids = torch.cat(
                (input_ids, zero_ids), dim=1
            )
            kwargs["input_ids"] = input_ids
            kwargs["output_attentions"] = False
        else:
            kwargs["input_ids"] = input_ids
            kwargs["output_attnetions"] = False

        # append 1s to attention_mask
        if (attention_mask is not None):
            # if accumilation is > 0 then n_bn_fusion = 0
            if self.accumilation > 0:
                # Create a tensor of ones with shape (B, n)
                ones_mask = torch.ones(
                    input_ids.size(0),
                    self.n_bn_fusion,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
            # otherwise, we have n_bn_fusion >= 0 and accumilation = 0
            else:
                # Create a tensor of ones with shape (B, n)
                ones_mask = torch.ones(
                    input_ids.size(0),
                    self.na_bn_fusion + self.nv_bn_fusion + self.n_bn_fusion,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
            # Concatenate the original attention_mask with the ones_mask
            attention_mask = torch.cat(
                (attention_mask, ones_mask), dim=1
            )
        kwargs["attention_mask"] = attention_mask
        return super().forward(**kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self):
        for layer in self._get_decoder_layers():
            layer.condition_audio(None)
            layer.condition_video(None)
            layer.condition_av(None)

class AttentionPattern(Enum):
    """Define different attention patterns"""

    CAUSAL = "causal"  # Lower triangular (autoregressive)
    FULL = "full"  # Full bidirectional attention
    NO_ATTEND = "no_attend"  # Cannot attend (blocked)
    CROSS_ONLY = "cross_only"  # Can attend to other modality but not self


@dataclass
class ModalityConfig:
    """Configuration for a single modality"""

    name: str
    length: int
    start_idx: int = 0  # Will be computed automatically

    def __post_init__(self):
        self.end_idx = self.start_idx + self.length

class MultimodalMaskBuilder:
    """Flexible mask builder for multimodal attention patterns"""

    def __init__(self, text_len: int, audio_len: int, video_len: int, av_len: int):
        """Initialize with text, audio, video lengths"""
        self.text_len = text_len
        self.audio_len = audio_len
        self.video_len = video_len
        self.av_len = av_len
        self.total_length = text_len + audio_len + video_len + av_len

        # Create modality configs
        self.modalities = {
            "text": ModalityConfig("text", text_len, 0),
            "audio": ModalityConfig("audio", audio_len, text_len),
            "video": ModalityConfig("video", video_len, text_len + audio_len),
            "audiovideo": ModalityConfig(
                "audiovideo", av_len, text_len + audio_len + video_len
            ),
        }

    def build_mask(
        self,
        attention_rules: Union[Dict[str, Dict[str, AttentionPattern]], str],
        device=None,
    ) -> torch.Tensor:
        """Build attention mask based on rules or preset name"""
        if isinstance(attention_rules, str):
            attention_rules = self.get_preset_rules(attention_rules)

        mask = torch.zeros(
            (self.total_length, self.total_length), dtype=torch.bool, device=device
        )

        for source_name, targets in attention_rules.items():
            source_mod = self.modalities[source_name]
            for target_name, pattern in targets.items():
                target_mod = self.modalities[target_name]
                self._apply_pattern(mask, source_mod, target_mod, pattern, device)

        return mask

    def _apply_pattern(
        self,
        mask: torch.Tensor,
        source: ModalityConfig,
        target: ModalityConfig,
        pattern: AttentionPattern,
        device=None,
    ):
        """Apply a specific attention pattern between two modalities"""
        s_start, s_end = source.start_idx, source.end_idx
        t_start, t_end = target.start_idx, target.end_idx

        if pattern == AttentionPattern.FULL:
            mask[s_start:s_end, t_start:t_end] = 1

        elif pattern == AttentionPattern.CAUSAL:
            # For causal: each token attends to itself and all previous tokens
            # This creates a lower triangular pattern across the ENTIRE sequence
            for i in range(s_start, s_end):
                for j in range(
                    t_start, min(t_end, i + 1)
                ):  # Only up to current position
                    mask[i, j] = 1

        elif pattern == AttentionPattern.NO_ATTEND:
            pass  # Keep as zeros

        elif pattern == AttentionPattern.CROSS_ONLY:
            if source.name != target.name:
                mask[s_start:s_end, t_start:t_end] = 1

    def get_preset_rules(
        self, preset_name: str
    ) -> Dict[str, Dict[str, AttentionPattern]]:
        """Get predefined attention rule presets"""
        presets = {
            "original": {
                "text": {
                    "text": AttentionPattern.CAUSAL,
                    "audio": AttentionPattern.NO_ATTEND,
                    "video": AttentionPattern.NO_ATTEND,
                    "audiovideo": AttentionPattern.NO_ATTEND,
                },
                "audio": {
                    "text": AttentionPattern.FULL,  # audio sees all text
                    "audio": AttentionPattern.CAUSAL,  # causal within audio
                    "video": AttentionPattern.NO_ATTEND,
                    "audiovideo": AttentionPattern.NO_ATTEND,
                },
                "video": {
                    "text": AttentionPattern.FULL,  # video sees all text
                    "audio": AttentionPattern.NO_ATTEND,
                    "video": AttentionPattern.CAUSAL,  # causal within video
                    "audiovideo": AttentionPattern.NO_ATTEND,
                },
                "audiovideo": {
                    "text": AttentionPattern.FULL,  # AV sees all text
                    "audio": AttentionPattern.NO_ATTEND,
                    "video": AttentionPattern.NO_ATTEND,
                    "audiovideo": AttentionPattern.CAUSAL,  # causal within AV
                },
            },
            "avseesall": {
                "text": {
                    "text": AttentionPattern.CAUSAL,
                    "audio": AttentionPattern.NO_ATTEND,
                    "video": AttentionPattern.NO_ATTEND,
                    "audiovideo": AttentionPattern.NO_ATTEND,
                },
                "audio": {
                    "text": AttentionPattern.FULL,
                    "audio": AttentionPattern.CAUSAL,
                    "video": AttentionPattern.NO_ATTEND,
                    "audiovideo": AttentionPattern.NO_ATTEND,
                },
                "video": {
                    "text": AttentionPattern.FULL,
                    "audio": AttentionPattern.NO_ATTEND,
                    "video": AttentionPattern.CAUSAL,
                    "audiovideo": AttentionPattern.NO_ATTEND,
                },
                "audiovideo": {
                    "text": AttentionPattern.FULL,
                    "audio": AttentionPattern.FULL,
                    "video": AttentionPattern.FULL,
                    "audiovideo": AttentionPattern.CAUSAL,
                },
            },
            "allcausal": {
                "text": {
                    "text": AttentionPattern.CAUSAL,
                    "audio": AttentionPattern.NO_ATTEND,
                    "video": AttentionPattern.NO_ATTEND,
                    "audiovideo": AttentionPattern.NO_ATTEND,
                },
                "audio": {
                    "text": AttentionPattern.FULL,
                    "audio": AttentionPattern.CAUSAL,
                    "video": AttentionPattern.NO_ATTEND,
                    "audiovideo": AttentionPattern.NO_ATTEND,
                },
                "video": {
                    "text": AttentionPattern.FULL,
                    "audio": AttentionPattern.CAUSAL,
                    "video": AttentionPattern.CAUSAL,
                    "audiovideo": AttentionPattern.NO_ATTEND,
                },
                "audiovideo": {
                    "text": AttentionPattern.FULL,
                    "audio": AttentionPattern.CAUSAL,
                    "video": AttentionPattern.CAUSAL,
                    "audiovideo": AttentionPattern.CAUSAL,
                },
                },
                "av_attends_only_av_a_v": {
                    "text": {
                        "text": AttentionPattern.CAUSAL,
                        "audio": AttentionPattern.NO_ATTEND,
                        "video": AttentionPattern.NO_ATTEND,
                        "audiovideo": AttentionPattern.NO_ATTEND,
                    },
                    "audio": {
                        "text": AttentionPattern.FULL,
                        "audio": AttentionPattern.CAUSAL,
                        "video": AttentionPattern.NO_ATTEND,
                        "audiovideo": AttentionPattern.NO_ATTEND,
                    },
                    "video": {
                        "text": AttentionPattern.FULL,
                        "audio": AttentionPattern.NO_ATTEND,
                        "video": AttentionPattern.CAUSAL,
                        "audiovideo": AttentionPattern.NO_ATTEND,
                    },
                    "audiovideo": {
                        "text": AttentionPattern.NO_ATTEND,
                        "audio": AttentionPattern.FULL,
                        "video": AttentionPattern.FULL,
                        "audiovideo": AttentionPattern.CAUSAL,
                    },
                },

        }
        return presets.get(preset_name, presets["original"])


class FlexibleCSAWrappedBlock(nn.Module):
    """Flexible multimodal attention block - ONLY modifies CSA mask, keeps everything else original"""

    def __init__(
        self,
        base_block,
        seq_len_text,
        seq_len_audio,
        seq_len_video,
        seq_len_audiovideo,
        layer_idx=None,
        default_pattern="original",
        attention_pattern=None,  # NEW: comes from config
    ):
        super().__init__()
        self.block = base_block  # Keep the ENTIRE original block

        # Store sequence lengths
        self.seq_len_text = seq_len_text
        self.seq_len_audio = seq_len_audio
        self.seq_len_video = seq_len_video
        self.seq_len_audiovideo = seq_len_audiovideo
        self.layer_idx = layer_idx
        self.default_pattern = default_pattern
        self.attention_pattern = attention_pattern

        # DEBUG: Print the parameters
        # print(f"🔍 FlexibleCSAWrappedBlock initialized with:")
        # print(f"   Text tokens: {seq_len_text}")
        # print(f"   Audio tokens: {seq_len_audio}")
        # print(f"   Video tokens: {seq_len_video}")
        # print(f"   Audiovisual tokens: {seq_len_audiovideo}")
        # print(
        #     f"   Total expected: {seq_len_text + seq_len_audio + seq_len_video + seq_len_audiovideo}"
        # )

        # Initialize mask builder
        self.mask_builder = MultimodalMaskBuilder(
            seq_len_text, seq_len_audio, seq_len_video, seq_len_audiovideo
        )

        # Cache for masks
        self._mask_cache = {}

        # Default pattern (your original behavior)

    def set_attention_pattern(self, pattern: Union[str, Dict]):
        """Set the default attention pattern"""
        self.default_pattern = pattern
        self._mask_cache.clear()

    def _get_current_modal_lengths(self, total_len):
        # Use ratios or real values depending on your input pipeline
        return {
            "T": self.seq_len_text,
            "A": self.seq_len_audio,
            "V": self.seq_len_video,
            "AV": self.seq_len_audiovideo,
        }

    def forward(self, *args, **kwargs):
        """Forward pass - accepts any arguments like the original block"""

        # Extract the hidden states (first argument)
     # --- 0) get hidden_states robustly (positional OR kwarg) ---
        x = kwargs.get("hidden_states", args[0] if len(args) else None)
        if x is None:
            raise RuntimeError("Expected hidden_states as first positional arg or in kwargs.")

        # Sometimes upstream accidentally passes a tuple (hs, ...) — unwrap
        if isinstance(x, (tuple, list)):
            x = x[0]

        # Must be [B, L, D]
        if x.dim() != 3:
            raise RuntimeError(
                f"[FlexibleCSAWrappedBlock L{self.layer_idx}] hidden_states must be [B,L,D], "
                f"got {tuple(x.shape)}"
            )
        # if self.layer_idx == 0:
        #     print(f"[WrappedBlock] incoming hidden_states: {tuple(x.shape)}")
        #     self._hs_shape_logged = True

        B, L, D = x.shape

        attention_pattern = self.attention_pattern
        pattern = (
            attention_pattern if attention_pattern is not None else self.default_pattern
        )

        # Validate sequence length
        expected_len = (
            self.seq_len_text
            + self.seq_len_audio
            + self.seq_len_video
            + self.seq_len_audiovideo
        )
        if L != expected_len:
            raise ValueError(f"Input sequence length {L} != expected {expected_len}")

        # 1) build a [L, L] boolean mask
        custom_mask = self._mask_cache.get(pattern)
        if custom_mask is None:
            custom_mask = self.mask_builder.build_mask(
                pattern, device=x.device
            )  # [L, L]
            self._mask_cache[pattern] = custom_mask

        # 2) reshape to [B, 1, L, L] so it can broadcast to heads
        #   FOR llama
        #custom_mask = custom_mask.unsqueeze(0).unsqueeze(1).expand(B, 1, L, L)
        #   FOR GPT
        custom_mask = custom_mask.unsqueeze(0).expand(B, -1, -1)

        # (2) AND with original key-padding on KEY side
        pad = kwargs.get("attention_mask", None)
        if pad is not None:
            # Accept a variety of HF shapes, convert to key-valid [B, L] (True = keep)
            if pad.dim() == 2:                 # [B, L] 1/0
                key_valid = pad.bool()
            elif pad.dim() == 3:               # [B, 1, L]
                key_valid = pad[:, -1, :].bool()
            elif pad.dim() == 4:               # [B, 1, 1, L] additive (0 or -inf)
                key_valid = (pad.squeeze(1).squeeze(1) == 0)
            else:
                raise RuntimeError(f"Unexpected attention_mask shape: {tuple(pad.shape)}")

            key_valid = key_valid.unsqueeze(1).expand(B, L, L)  # [B, L, L]
            custom_mask = custom_mask & key_valid

        # 4) Convert to additive mask [B, 1, L, L] (broadcastable to attn weights)
        add_mask = torch.where(
            custom_mask,
            torch.zeros(1, dtype=x.dtype, device=x.device),
            torch.full((1,), -1e4, dtype=x.dtype, device=x.device),
        ).unsqueeze(1)

        # 3) shove it into kwargs (rather than positional args)


        kwargs["attention_mask"] = add_mask
        #kwargs["attention_mask"] = custom_mask

        if self.training and self.layer_idx == 0:
            print_attention_mask_with_labels(
                add_mask[0, 0],  # shape [L, L]
                self.layer_idx,
                lengths=self._get_current_modal_lengths(x.shape[1]),
                pad_width=3,
            )

        # 4) call the original block
        return self.block(*args, **kwargs)

    def visualize_mask(
        self, pattern: Union[str, Dict] = None, save_path: Optional[str] = None
    ):
        """Visualize attention pattern"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for visualization")
            return

        pattern = pattern or self.default_pattern
        mask = self.mask_builder.build_mask(pattern)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask.numpy(), cmap="Blues", origin="upper")

        # Add boundaries
        ax.axhline(y=self.seq_len_text - 0.5, color="red", linewidth=2)
        ax.axhline(
            y=self.seq_len_text + self.seq_len_audio - 0.5, color="red", linewidth=2
        )
        ax.axhline(
            y=self.seq_len_text + self.seq_len_audio + self.seq_len_video - 0.5,
            color="red",
            linewidth=2,
        )

        ax.axvline(
            x=self.seq_len_text + self.seq_len_audio + self.seq_len_video - 0.5,
            color="red",
            linewidth=2,
        )
        ax.axvline(x=self.seq_len_text - 0.5, color="red", linewidth=2)
        ax.axvline(
            x=self.seq_len_text + self.seq_len_audio - 0.5, color="red", linewidth=2
        )

        # Add labels
        modalities = [
            ("Text", 0, self.seq_len_text),
            ("Audio", self.seq_len_text, self.seq_len_text + self.seq_len_audio),
            (
                "Video",
                self.seq_len_text + self.seq_len_audio,
                self.seq_len_text + self.seq_len_audio + self.seq_len_video,
            ),
            (
                "AV",
                self.seq_len_text + self.seq_len_audio + self.seq_len_video,
                self.seq_len_text
                + self.seq_len_audio
                + self.seq_len_video
                + self.seq_len_audiovideo,
            ),
        ]

        for name, start, end in modalities:
            mid = start + (end - start) // 2
            ax.text(-2, mid, name, rotation=90, va="center", ha="right", fontsize=12)
            ax.text(mid, -2, name, rotation=0, va="top", ha="center", fontsize=12)

        ax.set_title(f"Attention Pattern: {pattern}", fontsize=14, pad=20)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


# Global flag to ensure we print only once
_printed_mask_once = False


def print_attention_mask_with_labels(
    mask: torch.Tensor,
    layer_idx: int,
    lengths: dict,
    max_display_width: int = 600,
    pad_width: int = 2,
):
    global _printed_mask_once
    if _printed_mask_once:
        return

    mask_np = mask.int().cpu().numpy()
    L = mask_np.shape[0]

    # Generate modality labels
    labels = []
    modality_separators = []  # Index positions for dashed lines
    for tag in ["T", "A", "V", "AV"]:
        count = lengths.get(tag, 0)
        labels += [tag] * count
        if count > 0:
            modality_separators.append(len(labels))  # position after this modality

    assert len(labels) == L, f"Expected {L} labels but got {len(labels)}"

    print(f"\n🧠 Attention Mask (Layer {layer_idx}) — Shape: [{L} x {L}]")
    print("Queries = Rows | Keys = Columns\n")

    # --- Print Column Header ---
    header = "     "
    for i, lab in enumerate(labels):
        header += f"{lab:>2} "
        if i + 1 in modality_separators:
            header += "| "
    print(header.rstrip())

    # --- Print Each Row ---
    for i, row in enumerate(mask_np):
        row_label = labels[i]
        row_str = f"{row_label:>3}  "

        for j, val in enumerate(row):
            row_str += f"{val:>2} "
            if j + 1 in modality_separators:
                row_str += "| "

        print(row_str.rstrip())

        if i + 1 in modality_separators:
            # Print horizontal dashed line after current row
            line_length = len(row_str.rstrip())
            print("    " + "-" * line_length)

    _printed_mask_once = True






class MSALM(nn.Module):
    def __init__(self, config):
        """
        Args:
            config (dict): configuration for all other functionalities
        """
        super(MSALM, self).__init__()
        # useful args
        self.max_token_len = config.max_token_len
        self.n_bn_fusion = config.get("n_bn_fusion", -1)
        self.na_bn_fusion = config.get("na_bn_fusion", -1)
        self.nv_bn_fusion = config.get("nv_bn_fusion", -1)
        self.accumilation = config.get("accumilation", 0)
        self.modded_loss = config.get("modded_loss", False)
        self.use_ulgm = config.get("use_ulgm", False)
        self.lm_flavor = config['mmgpt'].get("type", 'llama')
        self.three_mod = (
            True  # three_mod used internally to destinguish each ablation type
        )

        # ------ language model ----------
        if 'chinese' in config.lm and not os.path.isabs(config.lm):
            lm_hf_path = f"uer/{config.lm}"
        else:
            lm_hf_path = config.lm
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            lm_hf_path,
            model_max_length=config.max_token_len,
            local_files_only=True,
        )
        self.tokenizer.padding_side = 'right'
        
        if 'chinese' in config.lm:
            # due to BERT tokenizer has pad and no eos
            self.tokenizer.eos_token = self.tokenizer.pad_token
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if config.get("use_bf16", False):
            llm_config = AutoConfig.from_pretrained(lm_hf_path)
            llm_config.output_attentions = False  # Ensure this is set to False
            llm_config.output_hidden_states = True
            # self.lang_encoder = AutoModelForCausalLM.from_pretrained(
            #     llm_config
            # )
            # self.lang_encoder = AutoModelForCausalLM.from_config(llm_config)
            self.lang_encoder = AutoModelForCausalLM.from_pretrained(
                lm_hf_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                return_dict_in_generate=True,
                output_hidden_states=True,
                use_cache=False,
            )
        else:
            if 'gpt' in self.lm_flavor:
                self.lang_encoder = AutoModelForCausalLM.from_pretrained(
                    lm_hf_path,
                    local_files_only=True,
                    output_hidden_states=True,
                    use_cache=False,
                )
            else:                
                self.lang_encoder = AutoModelForCausalLM.from_pretrained(
                    lm_hf_path,
                    local_files_only=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    use_cache=False,
                )
        
        # add mixing utilities
        extend_instance(self.lang_encoder, msaLMMixin)

        # add cross attentions
        self.lang_encoder.init_msalm(
            config["mmgpt"]["mm_layer"],
            config,
        )

        # if 'gpt' in self.lm_flavor:
        #     # manually deactivate layers that are not used
        # #     self.lang_encoder.transformer.wpe = None
        #     self.lang_encoder.transformer.drop = Identity()
        
        # task mapping
        # self.W_task = nn.Linear(
        #     config.mmgpt.d_mm + config.av_enc.d_enc_out,
        #     1
        # )
        ##############################################################################
        # original code here - before bn - uncomment afterwards
        ##########################################################################
        # self.W_task = nn.Linear(config.mmgpt.d_mm, 1)
        if self.use_ulgm:
            # task
            # main approach
            if self.three_mod:
                self.W_task = nn.Sequential(
                    nn.Linear(
                        4 * config.mmgpt.d_mm + config["av_enc"]["bimodal_encoder"]["d_enc_out"] + config["av_enc"]["audio_encoder"]["d_enc"] + config["av_enc"]["vision_encoder"]["d_enc"],
                        self.d_task,
                    ),
                    nn.BatchNorm1d(self.d_task),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_task, 1),
                )
            # ablation
            else:
                self.W_task = nn.Sequential(
                    nn.Linear(
                        3 * config.mmgpt.d_mm + 2 * config["av_enc"]["bimodal_encoder"]["d_enc_out"],
                        self.d_task,
                    ),
                    nn.BatchNorm1d(self.d_task),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_task, 1),
                )

            self.W_task_1 = nn.Linear(config.mmgpt.d_out, 1)
            # bn fusion
            self.W_bn = nn.Linear(config.mmgpt.d_mm, 1)
            # text map
            self.W_text = nn.Linear(config.mmgpt.d_mm, 1)
            # av map
            self.W_av = nn.Linear(config["av_enc"]["bimodal_encoder"]["d_enc_out"], 1)
            # audio map
            self.W_audio = nn.Linear(config["av_enc"]["audio_encoder"]["d_enc"], 1)
            # video map
            self.W_vision = nn.Linear(config["av_enc"]["vision_encoder"]["d_enc"], 1)

        elif self.modded_loss:
            self.d_task = config["mmgpt"].get("d_out", 64)

            # main approach
            if self.three_mod:
                #MODIFIY BELOW FOR NOT A,V TOKENS
                #+ config["av_enc"]["bimodal_encoder"]["d_enc_out"] + config["av_enc"]["audio_encoder"]["d_enc"] + config["av_enc"]["vision_encoder"]["d_enc"] for A,V encoidings
                self.W_task = nn.Sequential(
                    nn.Linear(
                        2 * config.mmgpt.d_mm + config["av_enc"]["bimodal_encoder"]["d_enc_out"]+config["av_enc"]["audio_encoder"]["d_enc"] + config["av_enc"]["vision_encoder"]["d_enc"],
                        self.d_task,
                    ),
                    nn.BatchNorm1d(self.d_task),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.d_task, 1),
                )
            # ablation
            # else: #elif self.two_mod
            # # NOT GONNA WORK, NEEDS FIXING
            #     self.W_task = nn.Sequential(
            #         nn.Linear(
            #             3 * config.mmgpt.d_mm + 2 * config["av_enc"]["d_enc_out"],
            #             self.d_task,
            #         ),
            #         nn.BatchNorm1d(self.d_task),
            #         nn.ReLU(inplace=True),
            #         nn.Linear(self.d_task, 1),
            #     )
            # else:
            #     self.W_task = nn.Sequential(
            #         nn.Linear(
            #             3 * config.mmgpt.d_mm + 2 * config["av_enc"]["d_enc_out"],
            #             self.d_task,
            #         ),
            #         nn.BatchNorm1d(self.d_task),
            #         nn.ReLU(inplace=True),
            #         nn.Linear(self.d_task, 1),
            #     )

            self.W_bn = nn.Linear(config.mmgpt.d_mm, 1)  # BN fusion uses fusion tokens
            self.W_text = nn.Linear(config.mmgpt.d_mm, 1)

            self.W_av = nn.Linear(config["av_enc"]["bimodal_encoder"]["d_enc_out"], 1)
            self.W_audio = nn.Linear(config["av_enc"]["audio_encoder"]["d_enc"], 1)
            self.W_vision = nn.Linear(config["av_enc"]["vision_encoder"]["d_enc"], 1)

        
        # ------ audiovisual encoder ----------
        # Initialize the AV encoder
        self.av_encoder = AV_Enc(config)

        # Handle pretrained loading
        av_config = config["av_enc"]
        if av_config.get("from_pretrained", False):
            print("----------------->>> Pretrained AudioVisual Encoder <<<<<----------------")

            single_path_bool = av_config.get("load_one_path", False)
            multiple_path_bool = av_config.get("load_multiple_paths", False)

            single_path = av_config.get("single_pretrained_path", None)
            # strict_loading = av_config.get("strict_loading", True)

            if single_path_bool and single_path:
                print("----------------->>> One path Pretrained AudioVisual Encoder <<<<<----------------")
                # optional override of prefixes if your checkpoint uses different names
                prefix_map = av_config.get("single_prefix_map", None)
                # example of prefix_map you can pass in config:
                # {
                #   "audio": ["audio_encoder.", "aud."],
                #   "video": ["vision_encoder.", "vis."],
                #   "bimodal": ["bimodal_encoder.", "bienc.", "av_fusion."]
                # }
                results = self.av_encoder.load_from_bienc_checkpoint(
                    ckpt_path=single_path,
                    strict=False,              # tolerate missing (e.g., no LN in BI_ENC)
                    verbose=True,
                    allow_overlap_copy=True    # copy overlapping slices for Linear/BN if dims changed
                )
                    
            elif multiple_path_bool:
                print("----------------->>> Multiple path Pretrained AudioVisual Encoder <<<<<----------------")
                # Check if individual pretrained paths are specified
                audio_config = av_config.get("audio_encoder", {})
                video_config = av_config.get("vision_encoder", {})
                bimodal_config = av_config.get("bimodal_encoder", {})
                
                audio_path = audio_config.get("audio_pretrained_path")
                video_path = video_config.get("video_pretrained_path")
                bimodal_path = bimodal_config.get("bimodal_pretrained_path")
                # joint_path = av_config.get("path_to_pretrained")
                
                # Determine loading strategy
                #individual_paths = any([audio_path, video_path, bimodal_path])

                # if individual_paths:
                #     print("Loading individual encoder weights...")
                    
                # Load individual encoders
                results = self.av_encoder.load_all_pretrained(
                    audio_path=audio_path,
                    video_path=video_path,
                    bimodal_path=bimodal_path,
                    strict=av_config.get("strict_loading", True)
                )
                
                # Report results
                for encoder_type, success in results.items():
                    status = "✓" if success else "✗"
                    print(f"{status} {encoder_type} encoder loading: {'SUCCESS' if success else 'FAILED'}")
                        
                # elif joint_path:
                #     print("Loading from jointly trained checkpoint...")
                #     self.av_encoder = AV_Enc.from_pretrained(joint_path, config)
                    
            else:
                print("ERROR: from_pretrained=True but no pretrained paths specified!")
                print("Please specify either:")
                print("  - 'path_to_pretrained' for jointly trained checkpoint")
                print("  - Individual paths: 'audio_pretrained_path', 'video_pretrained_path', 'bimodal_pretrained_path'")
                
        else:
            print("From Scratch AudioVisual Encoder Initialization")

        # add layer normalization
        self.use_lnorm = config.get("use_lnorm", False)
        if self.use_lnorm:
            print("------------------ Adding LNorm ------------------------")
            d_av = config["av_enc"]["bimodal_encoder"]["d_enc_out"]
            d_a = config["av_enc"]["audio_encoder"]["d_enc"]
            d_v = config["av_enc"]["vision_encoder"]["d_enc"]
            # self.RMS = nn.RMSNorm(d_av)
            self.LN_av = nn.LayerNorm(d_av)
            self.LN_a = nn.LayerNorm(d_a)
            self.LN_v = nn.LayerNorm(d_v)

        # Cross-Modal Consistency Mapping
        self.av_distil = config.get("av_distil", False)
        if self.av_distil:
            layers = []
            layers.append(nn.Linear(d_av, 1))
            self.av_dec = nn.Sequential(*layers)

        # avg pooling
        """
        maybe play with what we devide max_token_len with when averge pooling, 
        because pooling may be too aggressive
        """
        if self.accumilation > 0:
            kernel_size = config["max_token_len"] // (
                self.na_bn_fusion + self.nv_bn_fusion + self.accumilation
            )
            stride = kernel_size
            padding = 0
            self.av_pooling = nn.AvgPool1d(kernel_size, stride, padding)
        else:
            kernel_size = config["max_token_len"] // (
                self.na_bn_fusion + self.nv_bn_fusion + self.n_bn_fusion
            )
            stride = kernel_size
            padding = 0
            self.av_pooling = nn.AvgPool1d(kernel_size, stride, padding)

        
        # m3 - layer before classification
        self.p_m3 = config.get("p_m3", -1.0) 
        
    def forward(
        self,
        lang_x: torch.Tensor,
        audio_x: torch.Tensor,
        vision_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass of Flamingo.

        Args:
            lang_x (torch.Tensor): Language input ids
                shape (B, max_len)
            audio_x (torch.Tensor): audio input
                shape (B, max_len, D)
            video_x (torch.Tensor): video fused input
                shape (B, max_len, D)
            av_x (torch.Tensor): audiovisual fused input
                shape (B, max_len, D)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
        """
        assert (
            self.lang_encoder.initialized_msalm
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        a_x, v_x, av_x = self._encode_av(audio_x=audio_x, vision_x=vision_x)
        # print("calling _av_conditioning")
        self._av_conditioning(a_x, v_x, av_x)

        outputs = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            **kwargs
        )

        attns = outputs.attentions
        # attns is length = num_layers, each [batch, heads, seq_len, seq_len]


        lm_logits = outputs.logits
        last_hidden_states = outputs.hidden_states[-1]
        
        if self.use_ulgm:
            # last hidden state of the LM
            B, _, _ = last_hidden_states.size()
            with torch.no_grad():
                last_valid_indices = \
                                torch.sum(attention_mask, dim=1).long() - 1
            last_hidden_text = \
                last_hidden_states[
                    torch.arange(B, device=last_hidden_states.device),
                    last_valid_indices
                ]
            
            # modality masking
            if self.p_m3 > 0:
                # mask the language modality
                if self.p_m3 > random.random():
                    last_hidden_states = last_hidden_states * 0.0
            
            text_logits = self.W_text(last_hidden_text)

            # HERE order changes

            # Extract audio fusion tokens
            start_a = self.max_token_len
            end_a = start_a + self.na_bn_fusion
            audio_tokens = last_hidden_states[:, start_a:end_a, :]

            start_v = end_a
            end_v = start_v + self.nv_bn_fusion
            video_tokens = last_hidden_states[:, start_v:end_v, :]

            start_av = end_v
            if self.accumilation > 0:
                end_av = start_av + self.accumilation
                av_tokens = last_hidden_states[:, start_av:end_av, :]
            else:
                end_av = start_av + self.n_bn_fusion
                av_tokens = last_hidden_states[:, start_av:end_av, :]

            bn_tokens = torch.cat((audio_tokens, video_tokens, av_tokens), dim=1)
            bn_logits = self.W_bn(bn_tokens)

            # av output
            if a_x is not None:
                a_x = torch.mean(a_x, dim=1)
                a_logits = self.W_audio(a_x)
                # global fusion output
                avg_a_bn = torch.mean(audio_tokens, dim=1)  # average over fusion tokens
            else:
                a_logits = None

            if v_x is not None:
                v_x = torch.mean(v_x, dim=1)
                v_logits = self.W_vision(v_x)
                # global fusion output
                avg_v_bn = torch.mean(video_tokens, dim=1)  # average over fusion tokens
            else:
                v_logits = None

            if av_x is not None:
                av_x = torch.mean(av_x, dim=1)
                av_logits = self.W_av(av_x)
                # global fusion output
                avg_av_bn = torch.mean(av_tokens, dim=1)  # average over fusion tokens
            else:
                av_logits = None

            if (
                (av_logits is not None)
                and (a_logits is not None)
                and (v_logits is not None)
            ):
                self.three_mod = True
                task_outputs = self._task_map(
                    torch.cat(
                        (
                            avg_a_bn,
                            avg_v_bn,
                            avg_av_bn,
                            last_hidden_text,
                            a_x,
                            v_x,
                            av_x,
                        ),
                        dim=1,
                    )
                )
            elif (av_logits is not None) and (a_logits is not None):
                self.three_mod = False
                task_outputs = self._task_map(
                    torch.cat((avg_a_bn, avg_av_bn, last_hidden_text, a_x, av_x), dim=1)
                )
            elif (av_logits is not None) and (v_logits is not None):
                self.three_mod = False
                task_outputs = self._task_map(
                    torch.cat((avg_v_bn, avg_av_bn, last_hidden_text, v_x, av_x), dim=1)
                )
            elif (a_logits is not None) and (v_logits is not None):
                self.three_mod = False
                task_outputs = self._task_map(
                    torch.cat((avg_a_bn, avg_v_bn, last_hidden_text, a_x, v_x), dim=1)
                )

            return {
                "task_logits": task_outputs["task_logits"],
                "text_logits": text_logits,
                "a_logits": a_logits,
                "v_logits": v_logits,
                "av_logits": av_logits,
                "bn_logits": bn_logits,
                "lm_logits": None,
                "Feature_f": task_outputs["Feature_f"],
                "Feature_t": last_hidden_text,
                "Feature_a": avg_a_bn,
                "Feature_v": avg_v_bn,
                "Feature_bn": avg_av_bn,
            }         
        else:
            if self.modded_loss:
                # get tha last valid text representation
                # Linearly increasing weights for each batch (B, L)
                # linear_weights_batch = \
                #     torch.linspace(0, 1, self.max_token_len, device=self.args.device).repeat(B, 1)
                # row_sums = linear_weights_batch.sum(dim=1, keepdim=True)
                # dense_mask = linear_weights_batch / row_sums

                # last hidden state of the LM
                B, _, _ = last_hidden_states.size()
                with torch.no_grad():
                    last_valid_indices = torch.sum(attention_mask, dim=1).long() - 1
                last_hidden_text = last_hidden_states[
                    torch.arange(B, device=last_hidden_states.device),
                    last_valid_indices,
                ]
                # modality masking
                if self.p_m3 > 0:
                    # mask the language modality
                    if self.p_m3 > random.random():
                        last_hidden_states = last_hidden_states * 0.0

                text_logits = self.W_text(last_hidden_text)

                # Extract audio fusion tokens
                start_a = self.max_token_len
                end_a = start_a + self.na_bn_fusion
                audio_tokens = last_hidden_states[:, start_a:end_a, :]

                start_v = end_a
                end_v = start_v + self.nv_bn_fusion
                video_tokens = last_hidden_states[:, start_v:end_v, :]

                start_av = end_v
                if self.accumilation > 0:
                    end_av = start_av + self.accumilation
                    av_tokens = last_hidden_states[:, start_av:end_av, :]
                else:
                    end_av = start_av + self.n_bn_fusion
                    av_tokens = last_hidden_states[:, start_av:end_av, :]

                bn_tokens = torch.cat((audio_tokens, video_tokens, av_tokens), dim=1)
                #ONLY USE AV
                #bn_tokens=av_tokens
                bn_logits = self.W_bn(bn_tokens)

                # av output
                if a_x is not None:
                    a_x = torch.mean(a_x, dim=1)
                    a_logits = self.W_audio(a_x)
                    # global fusion output
                    avg_a_bn = torch.mean(
                        audio_tokens, dim=1
                    )  # average over fusion tokens
                else:
                    a_logits = None

                if v_x is not None:
                    v_x = torch.mean(v_x, dim=1)
                    v_logits = self.W_vision(v_x)
                    # global fusion output
                    avg_v_bn = torch.mean(
                        video_tokens, dim=1
                    )  # average over fusion tokens
                else:
                    v_logits = None

                if av_x is not None:
                    av_x = torch.mean(av_x, dim=1)
                    av_logits = self.W_av(av_x)
                    # global fusion output
                    avg_av_bn = torch.mean(
                        av_tokens, dim=1
                    )  # average over fusion tokens
                else:
                    av_logits = None

                """
                self.three_mod is set only when we have the main approach case, where we need different
                dimetntions for the task head because we have all 3 modalities. In any other ablation case
                we have 2 modalities, so the task head dimentions stay the same, hence a binary three_mod is enough
                """
                if (
                    (av_logits is not None)
                    and (a_logits is not None)
                    and (v_logits is not None)
                ):
                    self.three_mod = True
                    task_logits = self._task_map(
                        torch.cat(
                            (
                                # avg_a_bn,
                                # avg_v_bn,
                                avg_av_bn,
                                last_hidden_text,
                                a_x,
                                v_x,
                                av_x,
                            ),
                            dim=1,
                        )
                    )
                elif (av_logits is not None) and (a_logits is not None):
                    self.three_mod = False
                    task_logits = self._task_map(
                        torch.cat(
                            (avg_a_bn, avg_av_bn, last_hidden_text, a_x, av_x), dim=1
                        )
                    )
                elif (av_logits is not None) and (v_logits is not None):
                    self.three_mod = False
                    task_logits = self._task_map(
                        torch.cat(
                            (avg_v_bn, avg_av_bn, last_hidden_text, v_x, av_x), dim=1
                        )
                    )
                elif (a_logits is not None) and (v_logits is not None):
                    self.three_mod = False
                    task_logits = self._task_map(
                        torch.cat(
                            (avg_a_bn, avg_v_bn, last_hidden_text, a_x, v_x), dim=1
                        )
                    )
                else:
                    # ---- SINGLE-MODALITY FALLBACKS ----
                    if av_logits is not None:
                        task_logits = av_logits
                    elif a_logits is not None:
                        task_logits = a_logits
                    elif v_logits is not None:
                        task_logits = v_logits
                    else:
                        task_logits = text_logits

            return {
                "task_logits": task_logits,
                "text_logits": text_logits,
                "a_logits": a_logits,
                "v_logits": v_logits,
                "av_logits": av_logits,
                "bn_logits": bn_logits,
                "lm_logits": lm_logits,
            }

            # return lm_logits, task_logits, av_logits, bn_logits, text_logits

    def _task_map(self, h_last):
        # uses the final norm layer of the encoder (frozen)
        # h_last = self.lang_encoder.norm(h_last)
        if self.use_ulgm:
            h_fusion = self.W_task_0(h_last)
            fusion_logits = self.W_task_1(h_fusion)
            return {"Feature_f": h_fusion, "task_logits": fusion_logits}
        else:
            return self.W_task(h_last)

    def _av_conditioning(self, a_x, v_x, av_x):
        """
        Inject AV features into the language model
        """
        self.lang_encoder.clear_conditioned_layers()

        for layer in self.lang_encoder._get_decoder_layers():
            if self.na_bn_fusion > 0:  # bottleneck fusion
                layer.condition_audio(a_x)
            if self.nv_bn_fusion > 0:  # bottleneck fusion
                layer.condition_video(v_x)
            if (
                self.n_bn_fusion > 0
            ):  # only if we actually have encoded av information should we do this, in any other case - i.e accumilation - it is not valid
                layer.condition_av(av_x)
         
    
    def _encode_av(self, audio_x: torch.Tensor, vision_x: torch.Tensor):
        """
        Compute audiovisual latent representations and condition language model.
        Args:
            audio_x (torch.Tensor): Audio input
            vision_x (torch.Tensor): Vision input
            av_x (torch.Tensor): AudioVidual input
                shape (B, L_{m}, D_{m})
        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        # get new conditioning
        a_x, v_x, av_x = self.av_encoder(audio_x, vision_x)
        if self.use_lnorm:
            if a_x is not None:
                a_x = self.LN_a(a_x)
            if v_x is not None:
                v_x = self.LN_v(v_x)
            if av_x is not None:
                av_x = self.LN_av(av_x)
        return a_x, v_x, av_x
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x