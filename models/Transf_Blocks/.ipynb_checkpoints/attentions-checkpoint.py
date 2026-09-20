import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

import warnings
warnings.filterwarnings("ignore")
import time

# Copyright (c) Shanghai AI Lab. All rights reserved.

from typing import Sequence, Optional
import math, os

import logging
import numpy as np
import torch.utils.checkpoint as cp


from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed
from mmcls.models.backbones.base_backbone import BaseBackbone


# Alternative: Simplified WKV using attention-like mechanism
def wkv_attention_like(w, u, k, v):
    """
    Attention-like approximation of WKV computation
    This is faster but may not be as accurate as the true WKV
    """
    B, T, C = k.shape
    
    # Create attention weights using w and u
    # This is a simplified approximation
    decay_weights = torch.exp(-torch.abs(w).unsqueeze(0).unsqueeze(0))  # [1, 1, C]
    first_weights = torch.sigmoid(u).unsqueeze(0).unsqueeze(0)  # [1, 1, C]
    
    # Compute attention scores (simplified)
    scores = torch.matmul(k, k.transpose(-2, -1)) / math.sqrt(C)
    
    # Apply decay and first weights
    scores = scores * decay_weights.mean(dim=-1, keepdim=True)
    scores = scores + first_weights.mean(dim=-1, keepdim=True)
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, v)
    
    return output.squeeze(0).squeeze(0)



# Pure PyTorch implementation of bidirectional WKV
def bi_wkv_pytorch(w, u, k, v):
    """
    Pure PyTorch implementation of bidirectional WKV computation
    Args:
        w: spatial decay weights [B, T, C]
        u: spatial first weights [B, T, C] 
        k: key tensor [B, T, C]
        v: value tensor [B, T, C]
    Returns:
        output: [B, T, C]
    """
    B, T, C = k.shape
    device = k.device
    print(f"**********{dtype}")
    dtype = k.dtype
    
    # Forward pass
    wkv_forward = torch.zeros_like(v)
    state_forward = torch.zeros(B, C, device=device, dtype=dtype)
    
    for t in range(T):
        kt = k[:, t, :]  # [B, C]
        vt = v[:, t, :]  # [B, C]
        wt = w  # [C] broadcast to [B, C]
        ut = u  # [C] broadcast to [B, C]
        
        # WKV computation: wkv = (state + u * k * v) / (state_norm + u * k)
        wkv_num = state_forward + ut * kt * vt
        wkv_den = torch.abs(state_forward) + torch.abs(ut * kt) + 1e-8
        wkv_forward[:, t, :] = wkv_num / wkv_den
        
        # Update state: state = state * exp(-w) + k * v
        state_forward = state_forward * torch.exp(-torch.abs(wt)) + kt * vt
    
    # Backward pass (for bidirectional)
    wkv_backward = torch.zeros_like(v)
    state_backward = torch.zeros(B, C, device=device, dtype=dtype)
    
    for t in range(T-1, -1, -1):
        kt = k[:, t, :]  # [B, C]
        vt = v[:, t, :]  # [B, C]
        wt = w  # [C] broadcast to [B, C]
        ut = u  # [C] broadcast to [B, C]
        
        # WKV computation
        wkv_num = state_backward + ut * kt * vt
        wkv_den = torch.abs(state_backward) + torch.abs(ut * kt) + 1e-8
        wkv_backward[:, t, :] = wkv_num / wkv_den
        
        # Update state
        state_backward = state_backward * torch.exp(-torch.abs(wt)) + kt * vt
    
    # Combine forward and backward (average or other combination)
    output = (wkv_forward + wkv_backward) * 0.5
    
    return output


class WKV_PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v, use_attention_like=False):
        ctx.save_for_backward(w, u, k, v)
        ctx.use_attention_like = use_attention_like
        
        if use_attention_like:
            y = wkv_attention_like(w, u, k, v)
        else:
            y = bi_wkv_pytorch(w, u, k, v)
        
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        # For simplicity, use PyTorch's autograd
        # In practice, you might want to implement custom gradients
        return None, None, None, None, None


def RUN_PYTORCH(w, u, k, v, use_attention_like=True):
    """
    Pure PyTorch replacement for RUN_CUDA
    Args:
        w: spatial decay [C]
        u: spatial first [C]
        k: keys [B, T, C]
        v: values [B, T, C]
        use_attention_like: if True, use faster attention-like approximation
    """
    return WKV_PyTorch.apply(w, u, k, v, use_attention_like)

def q_shift(input, shift_pixel=1, gamma=1/4):
    assert gamma <= 1/4
    B, N, C = input.shape
    input = input.transpose(1, 2).reshape(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)

# Bidirectional Vision-RWKV (Bi-VRWKV) spatial mixer.
# NOTE: historically named "SwinAttention"; it is NOT Swin window attention.
# It is the linear-complexity bidirectional WKV recurrence (see bi_wkv_pytorch / RUN_PYTORCH).
# Kept as the `swa_attn1` module attribute for checkpoint/config compatibility.
class BiVRWKVAttention(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, init_mode='fancy', 
                 key_norm=False, with_cp=False, use_attention_like=True):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        self.use_attention_like = use_attention_like  # New parameter
        attn_sz = n_embd
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_v = None
            self.spatial_mix_r = None

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode=='fancy':
            with torch.no_grad(): # fancy init
                ratio_0_to_1 = (self.layer_id / (2 - 1)) # 0 to 1
                ratio_1_to_almost0 = (1.0 - (self.layer_id / 2)) # 1 to ~0
                
                # fancy time_decay
                decay_speed = torch.ones(self.n_embd)
                for h in range(self.n_embd):
                    decay_speed[h] = -5 + 8 * (h / (self.n_embd-1)) ** (0.7 + 1.3 * ratio_0_to_1)
                self.spatial_decay = nn.Parameter(decay_speed)

                # fancy time_first
                zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(self.n_embd)]) * 0.5)
                self.spatial_first = nn.Parameter(torch.ones(self.n_embd) * math.log(0.3) + zigzag)
                
                # fancy time_mix
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                self.spatial_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
        elif init_mode=='local':
            self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode=='global':
            self.spatial_decay = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_first = nn.Parameter(torch.zeros(self.n_embd))
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_v = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def jit_func(self, x):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        if self.shift_pixel > 0:
            xx = self.shift_func(x, self.shift_pixel, self.channel_gamma)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xv = x * self.spatial_mix_v + xx * (1 - self.spatial_mix_v)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xv = x
            xr = x

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x):
        def _inner_forward(x):
            B, T, C = x.size()
            self.device = x.device

            sr, k, v = self.jit_func(x)
            # Replace RUN_CUDA with RUN_PYTORCH
            x = RUN_PYTORCH(self.spatial_decay / T, self.spatial_first / T, k, v, self.use_attention_like)
            if self.key_norm is not None:
                x = self.key_norm(x)
            x = sr * x
            x = self.output(x)
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

# Channel-mixing activation variants for the ablation.
CHANNELMIX_ACTS = {
    'sq_relu': lambda k: torch.square(torch.relu(k)),
    'relu':    lambda k: torch.relu(k),
    'gelu':    lambda k: F.gelu(k),
    'swish':   lambda k: F.silu(k),
    'silu':    lambda k: F.silu(k),
}


class VRWKV_ChannelMix(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, hidden_rate=4, init_mode='fancy',
                 key_norm=False, with_cp=False, act='sq_relu'):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp
        if act not in CHANNELMIX_ACTS:
            raise ValueError(f"act must be one of {list(CHANNELMIX_ACTS)}, got {act!r}")
        self.act = act
        self.act_fn = CHANNELMIX_ACTS[act]
        self._init_weights(init_mode)
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_pixel > 0:
            self.shift_func = eval(shift_mode)
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self, init_mode):
        if init_mode == 'fancy':
            with torch.no_grad(): # fancy init of time_mix
                ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer)) # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        elif init_mode == 'local':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        elif init_mode == 'global':
            self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
            self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        else:
            raise NotImplementedError

    def forward(self, x):
        def _inner_forward(x):
            if self.shift_pixel > 0:
                xx = self.shift_func(x, self.shift_pixel, self.channel_gamma)
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
            else:
                xk = x
                xr = x

            k = self.key(xk)
            k = self.act_fn(k)   # ablatable: sq_relu (paper) | relu | gelu | swish
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class Block(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False, key_norm=False,
                 with_cp=False, use_attention_like=True):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = nn.Identity() #DropPath(drop_path) if drop_path > 0. else 
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, shift_pixel, init_mode,
                                   key_norm=key_norm, use_attention_like=use_attention_like)

        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, shift_pixel, hidden_rate,
                                   init_mode, key_norm=key_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp

    def forward(self, x):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x)))
                    x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x)))
                else:
                    x = x + self.drop_path(self.ln1(self.att(x)))
                    x = x + self.drop_path(self.ln2(self.ffn(x)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x)))
                    x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x)))
                else:
                    x = x + self.drop_path(self.att(self.ln1(x)))
                    x = x + self.drop_path(self.ffn(self.ln2(x)))
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class SparseCausalAttention(CrossAttention):
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0
        former_frame_index_1 = torch.arange(video_length) - 1
        former_frame_index_1[0] = 0
        former_frame_index_1[2] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index], key[:, former_frame_index_1]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index], value[:, former_frame_index_1]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
        


    
class SpatialTemporalAttention(CrossAttention):
    def forward_dense_attn(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = rearrange(key, "(b f) n d -> b f n d", f=video_length)
        key = key.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # (b f f n d)
        key = rearrange(key, "b f g n d -> (b f) (g n) d")

        value = rearrange(value, "(b f) n d -> b f n d", f=video_length)
        value = value.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # (b f f n d)
        value = rearrange(value, "b f g n d -> (b f) (g n) d")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, normal_infer=False):
        if normal_infer:
            return super().forward(
                hidden_states=hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask, 
                # video_length=video_length,
            )
        else:
            return self.forward_dense_attn(
                hidden_states=hidden_states, 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask, 
                video_length=video_length,
            )

class VRWKV(nn.Module):
    """
    Vision RWKV (Receptance Weighted Key Value) attention mechanism.
    A linear complexity alternative to CrossAttention for diffusion models.
    """
    
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.upcast_attention = upcast_attention
        self.scale = dim_head ** -0.5
        
        context_dim = context_dim or query_dim
        
        # RWKV specific parameters
        self.time_decay = nn.Parameter(torch.ones(heads, dim_head))
        self.time_first = nn.Parameter(torch.ones(heads, dim_head) * 0.3)
        
        # Receptance, Key, Value projections
        self.to_r = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=bias)
        
        # Time mixing parameters
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, query_dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, query_dim) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, query_dim) * 0.5)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim, bias=bias),
            nn.Dropout(dropout)
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(self.inner_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        # Initialize time decay between 0 and 1
        nn.init.uniform_(self.time_decay, 0.0, 1.0)
        
        # Initialize projections with xavier uniform
        nn.init.xavier_uniform_(self.to_r.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # Initialize output projection to zero for residual connection
        nn.init.zeros_(self.to_out[0].weight)
        if self.to_out[0].bias is not None:
            nn.init.zeros_(self.to_out[0].bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of VRWKV attention.
        
        Args:
            hidden_states: [batch, seq_len, dim] - Query states
            context: [batch, context_len, context_dim] - Key/Value states (if None, self-attention)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Use hidden_states as context for self-attention
        if context is None:
            context = hidden_states
        
        # Compute previous timestep for time mixing
        x_prev = torch.cat([
            torch.zeros_like(hidden_states[:, :1]),
            hidden_states[:, :-1]
        ], dim=1)
        
        context_prev = torch.cat([
            torch.zeros_like(context[:, :1]),
            context[:, :-1]
        ], dim=1)
        
        # Time mixing
        r = hidden_states * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        k = context * self.time_mix_k + context_prev * (1 - self.time_mix_k)
        v = context * self.time_mix_v + context_prev * (1 - self.time_mix_v)
        
        # Project to R, K, V
        r = self.to_r(r)  # [batch, seq_len, inner_dim]
        k = self.to_k(k)  # [batch, context_len, inner_dim]
        v = self.to_v(v)  # [batch, context_len, inner_dim]
        
        # Reshape for multi-head
        r = r.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        
        # Apply receptance (sigmoid gating)
        r = torch.sigmoid(r)
        
        # RWKV attention mechanism (linear complexity)
        if self.upcast_attention:
            r, k, v = r.float(), k.float(), v.float()
        
        # Compute WKV attention with time decay
        output = self._compute_wkv(r, k, v, mask)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.inner_dim
        )
        
        # Normalize for stability
        output = self.norm(output)
        
        # Output projection
        output = self.to_out(output)
        
        return output
    
    def _compute_wkv(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute WKV (Weighted Key-Value) attention with linear complexity.
        
        This is the core RWKV mechanism that replaces traditional QKV attention.
        """
        batch_size, heads, seq_len, dim_head = r.shape
        _, _, context_len, _ = k.shape
        
        # Simplified RWKV-style linear attention
        # Apply time decay weights
        time_decay = torch.sigmoid(self.time_decay).view(1, heads, 1, dim_head)
        time_first = torch.sigmoid(self.time_first).view(1, heads, 1, dim_head)
        
        # Compute attention weights with linear complexity
        # Using exponential decay for temporal modeling
        k = k * time_decay
        
        # Efficient WKV computation using einsum
        # This avoids the sequential loop and computes all positions at once
        if context_len == seq_len:
            # Self-attention case
            # Create causal mask for autoregressive attention
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=r.device))
            causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
            
            # Apply time-based decay
            time_diff = torch.arange(seq_len, device=r.device).view(1, -1) - torch.arange(seq_len, device=r.device).view(-1, 1)
            time_diff = torch.clamp(time_diff, min=0).float()
            decay_matrix = torch.exp(-time_diff * 0.1).unsqueeze(0).unsqueeze(0)  # Decay factor
            
            # Combine masks
            if mask is not None:
                attention_mask = causal_mask * mask.unsqueeze(1) * decay_matrix
            else:
                attention_mask = causal_mask * decay_matrix
            
            # Compute attention scores
            scores = torch.einsum('bhsd,bhtd->bhst', k, k) * self.scale
            scores = scores * attention_mask.to(scores.dtype)
            
            # Apply values
            output = torch.einsum('bhst,bhtd->bhsd', scores, v)
            
        else:
            # Cross-attention case
            # Simple linear attention without causal mask
            kv = torch.einsum('bhsd,bhsd->bhd', k, v).unsqueeze(2)  # Global key-value
            output = r * kv.expand(-1, -1, seq_len, -1)
        
        # Apply receptance gating
        output = output * r
        
        return output


def time_shift(x):
    """Shift along the temporal (frame) axis by one step, zero-padding the first
    frame. x: (B', T, C) where T is the number of frames. Returns x_{t-1}."""
    return torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)


class ControlledTemporalMixer(nn.Module):
    """
    Controlled temporal-mixing block comparison.

    Identical wrapper for every variant (LayerNorm -> Q/K/V projections ->
    sequence-mixing over the frame axis -> output projection -> LayerScale residual);
    only the core token-mixing-over-frames differs, so the comparison isolates the
    *mixing mechanism* rather than the surrounding pipeline:

      * 'full_attn'  : softmax attention over frames           -> O(T^2)   (Tune-A-Video)
      * 'wkv'        : bidirectional WKV linear attention       -> O(T) (linear), ours
      * 'linear_attn': kernel feature-map (elu+1) linear attn   -> O(T) (linear transformer)
      * 'ssm'        : minimal selective diagonal state-space    -> O(T) (Mamba-family)

    Operates on (B', T, C) with B' = batch*spatial_tokens, T = frames.
    """

    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.0, bias=False,
                 upcast_attention=False, mode="wkv", **kwargs):
        super().__init__()
        assert mode in ("full_attn", "wkv", "linear_attn", "ssm")
        self.mode = mode
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.ln = nn.LayerNorm(query_dim)
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, query_dim, bias=bias), nn.Dropout(dropout))
        self.gamma = nn.Parameter(torch.ones(query_dim) * 0.1)
        if mode == "wkv":
            self.time_decay = nn.Parameter(torch.zeros(heads))
            self.time_first = nn.Parameter(torch.zeros(heads))
        if mode == "ssm":
            # minimal selective diagonal SSM: input-dependent decay a_t = exp(dt_t * A)
            self.A_log = nn.Parameter(torch.zeros(heads, dim_head))   # A = -exp(A_log) in (-inf,0)
            self.dt_proj = nn.Linear(query_dim, heads, bias=True)
        for lin in [self.to_q, self.to_k, self.to_v, self.to_out[0]]:
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def _heads(self, t, B, T):
        return t.view(B, T, self.heads, self.dim_head).transpose(1, 2)  # B,H,T,D

    def _full_attn(self, q, k, v):
        a = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        return a @ v

    def _wkv(self, q, k, v):
        B, H, T, D = q.shape
        idx = torch.arange(T, device=q.device)
        dist = (idx[None, :] - idx[:, None]).abs().float()
        A = torch.exp(-dist[None] * torch.exp(self.time_decay).view(H, 1, 1))
        eye = torch.eye(T, device=q.device, dtype=torch.bool)[None]
        A = torch.where(eye, torch.exp(self.time_first).view(H, 1, 1).expand(H, T, T), A)
        k = k - k.amax(dim=2, keepdim=True)
        ek = torch.exp(k)
        num = torch.einsum('hst,bhtd->bhsd', A, ek * v)
        den = torch.einsum('hst,bhtd->bhsd', A, ek)
        return torch.sigmoid(q) * (num / (den + 1e-8))

    def _linear_attn(self, q, k, v):
        qf = F.elu(q) + 1.0
        kf = F.elu(k) + 1.0
        kv = torch.einsum('bhtd,bhte->bhde', kf, v)          # (B,H,D,D) global (non-causal)
        z = 1.0 / (torch.einsum('bhtd,bhd->bht', qf, kf.sum(dim=2)) + 1e-6)
        return torch.einsum('bhtd,bhde,bht->bhte', qf, kv, z)

    def _ssm(self, q, k, v, dt):
        # selective diagonal SSM scanned over T (T small). Input-dependent decay a_t
        # (from dt, A), input gate B_t = sigmoid(k) (selective), output gate C = sigmoid(q).
        B, H, T, D = v.shape
        A = -torch.exp(self.A_log).view(1, H, 1, D)           # (1,H,1,D) negative
        a = torch.exp(dt.unsqueeze(-1) * A)                   # (B,H,T,D) decay in (0,1), input-dependent
        inp = v * torch.sigmoid(k)                            # B_t: selective input gate
        s = torch.zeros(B, H, D, device=v.device, dtype=v.dtype)
        outs = []
        for t in range(T):
            s = a[:, :, t] * s + (1 - a[:, :, t]) * inp[:, :, t]
            outs.append(s)
        y = torch.stack(outs, dim=2)
        return torch.sigmoid(q) * y                           # C_t: output gate

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        x = hidden_states
        B, T, _ = x.shape
        h = self.ln(x)
        q = self._heads(self.to_q(h), B, T).float()
        k = self._heads(self.to_k(h), B, T).float()
        v = self._heads(self.to_v(h), B, T).float()
        if self.mode == "full_attn":
            o = self._full_attn(q, k, v)
        elif self.mode == "wkv":
            o = self._wkv(q, k, v)
        elif self.mode == "linear_attn":
            o = self._linear_attn(q, k, v)
        else:  # ssm
            dt = F.softplus(self.dt_proj(h)).transpose(1, 2)  # (B,H,T)
            o = self._ssm(q, k, v, dt.float())
        o = o.transpose(1, 2).reshape(B, T, self.inner_dim).to(x.dtype)
        return x + self.gamma * self.to_out(o)


class VRWKVTemporalAttention(nn.Module):
    """
    Vision-RWKV-style temporal mixing block (drop-in for the temporal attention).

    Operates on a per-spatial-token sequence over frames, shape (B', T, C) with
    B' = batch * spatial_tokens and T = frames. It is a self-contained residual
    block with two sub-modules, following the canonical RWKV time-mix / channel-mix
    decomposition adapted to the temporal axis of video:

      time-mix    : x <- x + gamma_t * WKV( shift-interp(x; mu) )
      channel-mix : x <- x + gamma_c * FFN( shift-interp(x; mu_c) )

    where the shift-interpolation blends each frame token with its previous-frame
    counterpart, channel-wise:

        x_tilde = mu  (.) x_t + (1 - mu ) (.) x_{t-1}        # time-mix    (paper: mu)
        x_tilde = mu_c(.) x_t + (1 - mu_c) (.) x_{t-1}       # channel-mix (paper: mu_c)

      * mu   in R^{1x1x1x1xC}  : learnable channel-wise interpolation parameter.
      * mu_c in R^{1x1x1x1xC}  : a *separate* learnable parameter for channel
        mixing that lets the model adaptively control temporal smoothness in the
        channel domain (mu_c -> 1: each channel keeps the current frame; mu_c -> 0:
        it averages with the previous frame -> stronger temporal smoothing).

    Key fix vs. the previous implementation: the output paths are gated by a small
    *learnable* LayerScale (gamma_t, gamma_c) initialised to 0.1 rather than a hard
    zero-init on the output projection. A zero-initialised output projection makes
    the gradient w.r.t. the interpolation parameters identically zero, which froze
    mu / mu_c at their initial 0.5 in the old model. The LayerScale keeps the branch
    near-identity at the start (training stability when inflating 2D SD weights)
    while still letting mu and mu_c receive gradient and actually learn.
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        hidden_rate: int = 4,
        **kwargs  # Catch any additional arguments
    ):
        super().__init__()
        C = query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.upcast_attention = upcast_attention

        # block-internal norms (this module owns its residuals)
        self.ln_t = nn.LayerNorm(C)
        self.ln_c = nn.LayerNorm(C)

        # --- time-mix (RWKV temporal attention) ---
        # mu: channel-wise interpolation parameter, R^{1x1x1x1xC} (stored as (1,1,C)
        #     since the temporal tensor here is (B', T, C)). We parameterize it via a
        #     logit and take mu = sigmoid(mu_logit) so mu stays a true interpolation
        #     weight in (0, 1); mu_logit init 0 -> mu init 0.5.
        self.mu_logit = nn.Parameter(torch.zeros(1, 1, C))
        self.to_r = nn.Linear(C, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(C, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(C, self.inner_dim, bias=bias)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, C, bias=bias), nn.Dropout(dropout))
        # per-head bidirectional decay / current-frame bonus for the Bi-WKV kernel
        self.time_decay = nn.Parameter(torch.zeros(heads))
        self.time_first = nn.Parameter(torch.zeros(heads))

        # --- channel-mix (RWKV FFN) ---
        # mu_c: separate channel-wise interpolation parameter, R^{1x1x1x1xC},
        #       likewise parameterized via a logit -> mu_c = sigmoid(mu_c_logit) in (0,1).
        self.mu_c_logit = nn.Parameter(torch.zeros(1, 1, C))
        hidden = C * hidden_rate
        self.cm_key = nn.Linear(C, hidden, bias=bias)
        self.cm_value = nn.Linear(hidden, C, bias=bias)
        self.cm_receptance = nn.Linear(C, C, bias=bias)

        # LayerScale gates: small but non-zero -> stable start + gradient to mu / mu_c
        self.gamma_t = nn.Parameter(torch.ones(C) * 0.1)
        self.gamma_c = nn.Parameter(torch.ones(C) * 0.1)

        self._init_weights()

    def _init_weights(self):
        for lin in [self.to_r, self.to_k, self.to_v, self.to_out[0],
                    self.cm_key, self.cm_value, self.cm_receptance]:
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def _bi_wkv(self, r, k, v):
        """Bidirectional WKV over the (small) temporal axis. r,k,v: (B,H,T,D)."""
        B, H, T, D = r.shape
        idx = torch.arange(T, device=r.device)
        dist = (idx[None, :] - idx[:, None]).abs().float()          # (T,T)
        w = torch.exp(self.time_decay).view(H, 1, 1)                # positive decay rate
        A = torch.exp(-dist[None] * w)                             # (H,T,T) off-diagonal decay
        eye = torch.eye(T, device=r.device, dtype=torch.bool)[None]
        A = torch.where(eye, torch.exp(self.time_first).view(H, 1, 1).expand(H, T, T), A)
        # numerically-stable exp(k) (invariant to per-(B,H,D) shift)
        k = k - k.amax(dim=2, keepdim=True)
        ek = torch.exp(k)                                          # (B,H,T,D)
        num = torch.einsum('hst,bhtd->bhsd', A, ek * v)
        den = torch.einsum('hst,bhtd->bhsd', A, ek)
        return torch.sigmoid(r) * (num / (den + 1e-8))

    def _time_mix(self, x):
        B, T, _ = x.shape
        mu = torch.sigmoid(self.mu_logit)                        # mu in (0,1)
        xm = x * mu + time_shift(x) * (1 - mu)                   # mu interpolation
        r, k, v = self.to_r(xm), self.to_k(xm), self.to_v(xm)
        r = r.view(B, T, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(B, T, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, T, self.heads, self.dim_head).transpose(1, 2)
        o = self._bi_wkv(r.float(), k.float(), v.float())        # WKV in fp32 for stability
        o = o.transpose(1, 2).reshape(B, T, self.inner_dim).to(x.dtype)
        return self.to_out(o)

    def _channel_mix(self, x):
        mu_c = torch.sigmoid(self.mu_c_logit)                    # mu_c in (0,1)
        xm = x * mu_c + time_shift(x) * (1 - mu_c)              # mu_c interpolation
        k = torch.square(torch.relu(self.cm_key(xm)))            # squared-ReLU FFN
        return torch.sigmoid(self.cm_receptance(xm)) * self.cm_value(k)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        # hidden_states: (B', T, C); this block applies its own norms + residuals.
        x = hidden_states
        x = x + self.gamma_t * self._time_mix(self.ln_t(x))
        x = x + self.gamma_c * self._channel_mix(self.ln_c(x))
        return x


