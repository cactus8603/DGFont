from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import scipy.io as io
import math
import numpy as np
import sys

from blocks import LinearBlock, Conv2dBlock, ResBlocks
from modulated_deform_conv import *

class Generator(nn.Module):
    def __init__(self, img_size=80, sty_dim=64, n_res=2, use_sn=False):
        super(Generator, self).__init__()

        self.nf = 64 
        self.nf_mlp = 256
        self.decoder_norm = 'adain'
        
        self.adaptive_param_getter = get_num_adain_params
        self.adaptive_param_assign = assign_adain_params

        n_downs = 2
        self.cnt_encoder = ContentEncoder(self.nf, n_downs, n_res, 'in', 'relu', 'reflect')
        self.decoder = Decoder(nf_dec, sty_dim, n_downs, n_res, self.decoder_norm, self.decoder_norm, 'relu', 'reflect', use_sn=use_sn)
        self.mlp = MLP(sty_dim, self.adaptive_param_getter(self.decoder), self.nf_mlp, 3, 'none', 'relu')

        self.apply(weights_init('kaiming'))

        def forward(self, x_src, s_ref):
            c_src, skip1, skip2 = self.cnt_encoder(x_src)
            x_out = self.decode(c_src, s_ref, skip1, skip2)
            return x_out

        def decode(self, cnt, sty, skip1, skip2):
            adapt_params = self.mlp(sty)
            self.adaptive_param_assign(adapt_params, self.decoder)
            out = self.decoder(cnt, skip1, skip2)
            return out

        def _initialize_weights(self, mode='fan_in')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self, nf_dec, sty_dim, n_downs, n_res, res_norm, dec_norm, act, pad, use_sn=False):
        super(Decoder, self).__init__()
        
        nf = nf_dec
        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, nf, res_norm, act, pad, use_sn=use_sn))
        
        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(nf, nf//2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))
        nf //= 2

        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(2*nf, nf//2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))
        nf //= 2

        self.model.append(Conv2dBlock(2*nf, 3, 7, 1, 3, norm='none', act='tanh', pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)
        self.dcn = ModulatedDeformConvPack(64, 64, kernel_size=(3, 3), stride=1, padding=1, groups=1, deformable_groups=1, double=True).cuda()

    def forward(self, x, skip1, skip2):
        output = x
        for i in range(len(self.model)):
            output = self.model[i](output)

            if i == 2:
                deformable_concat = torch.cat((output, skip2), dim=1)
                concat_pre, offset2 = self.dcn2(deformable_concat, skip2)
                output = torch.cat((concat_pre, output), dim=1)

            if i == 4:
                deformable_concat = torch.cat((output, skip1), dim=1)
                concat_pre, offset1 = self.dcn, (deformable_concat, skip1)
                output = torch.cat((concat_pre, output), dim=1)

        offset_sum1 = torch.mean(torch.abs(offset1))
        offset_sum2 = torch.mean(torch.abs(offset2))
        offset_sum = (offset_sum1 + offset_sum2) / 2

        return output, offset_sum

class ContentEncoder(nn.Module):
    def __init__(self, nf_cnt, n_downs, n_res, norm, act, pad, use_sn=False):
        super(ContentEncoder, self).__init__()

        nf = nf_cnt

        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, 256, norm=norm, act=act, pad_type=pad, use_sn=use_sn)
        self.model = nn.Sequential(*self.model)

        self.dcn1 = ModulatedDeformConvPack(3, 64, kernel_size=(7, 7), stride=1, padding=3, groups=1, deformable_groups=1).cuda()
        self.dcn2 = modulated_deform_conv.ModulatedDeformConvPack(64, 128, kernel_size=(4, 4), stride=2, padding=1, groups=1, deformable_groups=1).cuda()
        self.dcn3 = modulated_deform_conv.ModulatedDeformConvPack(128, 256, kernel_size=(4, 4), stride=2, padding=1, groups=1, deformable_groups=1).cuda()

        self.IN1 = nn.InstanceNorm2d(64)
        self.IN2 = nn.InstanceNorm2d(128)
        self.IN3 = nn.InstanceNorm2d(256)
        self.activation = nn.ReLU(inplace=True)

