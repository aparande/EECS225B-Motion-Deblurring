import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class ConvBlock(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class FC(nn.Module):
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc))
        self.activation = activation() if activation is not None else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class SliceLayer(nn.Module):
    def __init__(self):
        super(SliceLayer, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)
        return coeff.squeeze(2)

class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):
        '''
        For each output channel, multipy the full res input by channels c'+4c in the sliced feature map 
        and sum over all channels to get output channel c.
        '''
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)

class GuidanceMap(nn.Module):
    def __init__(self, params=None):
        super(GuidanceMap, self).__init__()
        self.params = params
        # TODO: How does this guidemap work? Why can we just use a conv block
        # The original paper does not specify 16, and it uses sigmoid instead of Tanh
        self.conv1 = ConvBlock(3, params['guide_complexity'], kernel_size=1, padding=0, batch_norm=params['batch_norm'])
        self.conv2 = ConvBlock(params['guide_complexity'], 1, kernel_size=1, padding=0, activation=nn.Tanh)
        #self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Sigmoid)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class ComputeCoeffs(nn.Module):
    def __init__(self, params=None):
        super(ComputeCoeffs, self).__init__()

        self.lb = params['luma_bins'] # Number of 
        self.cm = params['channel_multiplier']
        self.sb = params['spatial_bin']
        self.bn = params['batch_norm']
        self.nsize = params['net_input_size']

        # Build the low-level features
        splat_num = int(np.log2(self.nsize / self.sb))
        self.lowlevel_features = nn.ModuleList()

        low_level_channel_num = 3
        for i in range(splat_num):
            use_bn = self.bn if i > 0 else False
            layer = ConvBlock(low_level_channel_num, self.cm * (2 ** i) * self.lb, kernel_size=3, stride=2, batch_norm=use_bn)
            self.lowlevel_features.append(layer)
            low_level_channel_num = self.cm * (2 ** i) * self.lb

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(low_level_channel_num, 8 * self.cm * self.lb, kernel_size=3, batch_norm=self.bn))
        self.local_features.append(ConvBlock(8*self.cm*self.lb, 8*self.cm*self.lb, kernel_size=3, use_bias=False))

        # Global Features
        global_num = int(np.log2(self.sb/4))
        global_channel_num = low_level_channel_num
        self.global_features_conv = nn.ModuleList()
        for i in range(global_num):
            layer = ConvBlock(global_channel_num, self.cm * 8 * self.lb, kernel_size=3, stride=2, batch_norm=self.bn)
            self.global_features_conv.append(layer)
            global_channel_num = self.cm * 8 * self.lb

        self.global_features_fc = nn.ModuleList()

        n_total = splat_num + global_num
        fc_feature_num = global_channel_num * (self.nsize / (2 ** n_total)) ** 2
        self.global_features_fc.append(FC(fc_feature_num, 32 * self.cm * self.lb, batch_norm=self.bn))
        self.global_features_fc.append(FC(32 * self.cm * self.lb, 16 * self.cm * self.lb, batch_norm=self.bn))
        self.global_features_fc.append(FC(16 * self.cm * self.lb, 8 * self.cm * self.lb, activation=None, batch_norm=self.bn))
        
        self.conv_out = ConvBlock(8 * self.cm * self.lb, self.lb * 3 * 4, 1, padding=0, activation=None)
   
    def forward(self, lowres_input):
        bs = lowres_input.shape[0]

        x = lowres_input
        for layer in self.lowlevel_features:
            x = layer(x)
        splat_features = x
        
        for layer in self.global_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x

        x = splat_features
        for layer in self.local_features:
            x = layer(x)        
        local_features = x

        fusion_grid = local_features
        fusion_global = global_features.view(bs, 8 * self.cm * self.lb, 1, 1)
        fusion = nn.ReLU()(fusion_grid + fusion_global)

        x = self.conv_out(fusion)
        s = x.shape
        x = x.view(bs, 3 * 4, self.lb, self.sb, self.sb) # B x Coefs x Luma x Spatial x Spatial
        return x

class HDRPointwiseNN(nn.Module):

    def __init__(self, params):
        super(HDRPointwiseNN, self).__init__()
        self.coeffs = ComputeCoeffs(params=params)
        self.guide = GuidanceMap(params=params)
        self.slice = SliceLayer()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, lowres, fullres):
        coeffs = self.coeffs(lowres)
        guide = self.guide(fullres)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)
        return out


#########################################################################################################


    
