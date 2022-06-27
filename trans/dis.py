import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import scipy.signal

from utils import *

# original:https://github.com/VITA-Group/TransGAN/blob/master/models_search/Celeba256_dis.py
# lack function:
# from models_search.ViT_helper import DropPath, to_2tuple, trunc_normal_
# from models_search.diff_aug import DiffAugment

# from utils.utils import make_grid, save_image

# from models_search.ada import *
# import scipy.signal
# from torch_utils.ops import upfirdn2d

class matmul(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2):
        x = x1@x2
        
        return x

# activative function which depends on bernoulli distribution
def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def leakyrelu(x):
    return F.leaky_relu_(x, 0.2)


class CustomAct(nn.Moudle):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer == gelu
        elif act_layer == "leakyrelu":
            self.act_layer == leakyrelu


class Mlp(nn.Module):
    def __init__(self, in_feature, hidden_feature=None, out_feature=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_feature = out_feature or in_feature
        hidden_feature = hidden_feature or in_feature
        self.fc1 = nn.Linear(in_feature, hidden_feature)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear()(hidden_feature, out_feature)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., is_mask=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.is_mask = is_mask

    def forward(self, x):
        B, N, C = x.shape
        # need to check what mask means
        if self.is_mask == 1:
            H = W = int(math.sqrt(N))
            image = x.view(B, H, W, C).view(B*H, W, C)
            qkv = self.qkv(image).reshape(B*H, W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = self.mat(attn, v).transpose(1,2)
            x = x.reshape(B*H, W, C).view(B, H, W, C).view(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        elif self.is_mask == 2:
            H = W = int(math.sqrt(N))
            # add permute(0,2,1,3)
            image = x.view(B, H, W, C).permute(0,2,1,3).view(B*H, W, C)
            qkv = self.qkv(image).reshape(B*H, W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # add permute(0,2,1,3)
            x = self.mat(attn, v).transpose(1,2).reshape(B*W, H, C).view(B, W, H, C).permute(0,2,1,3).reshape(B, N, C)
            x = x.reshape(B*H, W, C).view(B, H, W, C).view(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = self.mat(attn, v).transpose(1,2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        
        return x

class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)

class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm == nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm == nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm == PixelNorm(dim)
    
    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0,2,1)).permute(0,2,1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous.view(-1, window_size, window_size, C)
    
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H*W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    
    return x 

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
    drop=0., attn_drop=0., drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_feature=dim, hidden_feature=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x