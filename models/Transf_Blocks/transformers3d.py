# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

from .attentions import BiVRWKVAttention, SparseCausalAttention, SpatialTemporalAttention, VRWKVTemporalAttention, VRWKV_ChannelMix, ControlledTemporalMixer


# --- Ablation hook: optional VRWKV channel-mixing ---
# Disabled by default so existing checkpoints (which contain no channel-mix
# parameters) load unchanged. Call configure_channel_mix(...) BEFORE building the
# U-Net to enable it and pick the activation (sq_relu | relu | gelu | swish).
_CHANNEL_MIX = {"enabled": False, "act": "sq_relu"}


def configure_channel_mix(enabled: bool, act: str = "sq_relu"):
    _CHANNEL_MIX["enabled"] = bool(enabled)
    _CHANNEL_MIX["act"] = act


# --- Sensitivity hook: spatial token-shift hyperparameters of the
# Bi-VRWKV spatial mixer. Defaults match the paper (shift_pixel=1, channel_gamma=1/4).
# Call configure_spatial_mix(...) BEFORE building the U-Net to override them. ---
_SPATIAL_MIX = {"shift_pixel": 1, "channel_gamma": 0.25}


def configure_spatial_mix(shift_pixel: int = 1, channel_gamma: float = 0.25):
    _SPATIAL_MIX["shift_pixel"] = int(shift_pixel)
    _SPATIAL_MIX["channel_gamma"] = float(channel_gamma)


# --- Architecture mode (efficiency redesign) ---
# The original model runs BOTH SD's quadratic attn1 AND the linear Bi-VRWKV mixer
_ARCH = {"spatial_mode": "both"}


def configure_arch(spatial_mode: str = "both"):
    assert spatial_mode in ("both", "linear_only", "quad_only")
    _ARCH["spatial_mode"] = spatial_mode


# --- Controlled temporal-mechanism benchmark---
# When set, the temporal block (attn_temp) is replaced by ControlledTemporalMixer
# with the chosen sequence-mixing core, so the comparison isolates the mechanism:
#   None (default) -> the deployed VRWKVTemporalAttention
#   'full_attn' | 'wkv' | 'linear_attn' | 'ssm' -> controlled mixer
_TEMPORAL = {"mixer": None}


def configure_temporal(mixer):
    assert mixer in (None, "full_attn", "wkv", "linear_attn", "ssm")
    _TEMPORAL["mixer"] = mixer

from einops import rearrange, repeat


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_swin_attention: bool = True,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_swin_attention = use_swin_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # SC-Attn
        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        
        # Bi-VRWKV spatial mixer (attribute name `swa_attn1` kept for checkpoint/config compatibility)
        self.swa_attn1 = BiVRWKVAttention(
            n_embd=dim, n_layer=1, layer_id=1, init_mode='local',
            shift_pixel=_SPATIAL_MIX["shift_pixel"], channel_gamma=_SPATIAL_MIX["channel_gamma"],
        )

        # Optional VRWKV channel-mixing (paper eq. 26) for the activation ablation.
        # Off by default -> no extra params, existing checkpoints unaffected.
        self.use_channel_mix = _CHANNEL_MIX["enabled"]
        if self.use_channel_mix:
            self.channel_mix = VRWKV_ChannelMix(
                n_embd=dim, n_layer=1, layer_id=1, init_mode='local', act=_CHANNEL_MIX["act"],
            )
            self.norm_cm = nn.LayerNorm(dim)
            # Zero-init the output projection so this (spatial-block) channel-mix
            # starts as an exact identity residual. Without it the freshly-added module
            # injects noise from step 0 and a short fine-tune just recovers from it;
            nn.init.zeros_(self.channel_mix.value.weight.data)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn (default: deployed VRWKV temporal; or a controlled mixer)
        if _TEMPORAL["mixer"] is not None:
            self.attn_temp = ControlledTemporalMixer(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                mode=_TEMPORAL["mixer"],
            )
        else:
            self.attn_temp = VRWKVTemporalAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        # NB: the temporal block is now a self-contained residual module with its own
        # norms and a learnable LayerScale (gamma_t / gamma_c) for stability, so we do
        # NOT zero-init its output projection — doing so would zero the gradient to the
        # mu / mu_c interpolation parameters and freeze them at init (see attentions.py).
        self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            self.swa_attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        _mode = _ARCH["spatial_mode"]
        # Quadratic SD spatial self-attention (skipped in linear_only mode).
        if _mode != "linear_only":
            if self.only_cross_attention:
                hidden_states = (
                    self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
                )
            else:
                hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states

        # Linear Bi-VRWKV spatial mixer (skipped in quad_only mode).
        if _mode != "quad_only":
            hidden_states = self.swa_attn1(norm_hidden_states) + hidden_states

        # Channel-mixing residual (paper eq. 27: Z = X + Y_spatial + CM); ablation-gated.
        if self.use_channel_mix:
            hidden_states = self.channel_mix(self.norm_cm(hidden_states)) + hidden_states


        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )
            print(f"******hidden_states_shaoe : {hidden_states.shape} hidden_states_shaoe : {norm_hidden_states.shape}  and in_hidden_states {encoder_hidden_states.shape}", file=open("hidden_states2.txt",'a'))

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention: self-contained Vision-RWKV temporal block
        # (internal norms + time-mix(mu) + channel-mix(mu_c) + residuals).
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        hidden_states = self.attn_temp(hidden_states)
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
