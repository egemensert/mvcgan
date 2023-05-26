import torch
import numpy as np
import torch.nn as nn
from kornia.filters import filter2d
from custom_layers import EqualizedConv2d, Upscale2d

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, input_dim, output_dim, eps=1e-8) -> None:
        super(AdaptiveInstanceNorm, self).__init__()

        self.eps = eps
        self.style_modulator = nn.Linear(input_dim, 2*output_dim)
        self.output_dim = output_dim

        with torch.no_grad():
            self.style_modulator.weight *= 0.25
            self.style_modulator.bias.data.fill_(0)

    def forward(self, x, y):
        B, C, W, H = x.size()

        x_temp = x.view(B, C, -1)
        mean = x_temp.mean(dim=2).view(B, C, 1, 1)
        mean2 = mean * mean
        x2 = (x_temp * x_temp).mean(dim=2).view(B, C, 1, 1)
        x_var = torch.clamp(x2 - mean2, min=0)
        x_std_inv = torch.rsqrt(x_var + self.eps)

        x_norm = (x - mean) * x_std_inv 

        y_style = self.style_modulator(y)
        y_scale = y_style[:, :self.output_dim].view(B, self.output_dim, 1, 1)
        y_loc = y_style[:, self.output_dim:].view(B, self.output_dim, 1, 1)

        # output_dim should match x's channel size.
        return y_scale * x_norm + y_loc
    
class UnitConv(nn.Module):
    def __init__(self):
        super(UnitConv, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        self.conv.weight.data.fill_(0)

    def forward(self, x):
        return self.conv(x)
    
class Blur(nn.Module):
    ''' Retrieved from: https://github.com/Xuanmeng-Zhang/MVCGAN/blob/7791afc36ec3c257120e582470962b513f8a77dc/generators/refinegan.py'''
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)
    
class Scaler(nn.Module):
    def __init__(self, last_dim, new_dim, mapping_dim, leak) -> None:
        super(Scaler, self).__init__()
        self.conv1 = EqualizedConv2d(
            last_dim,
            new_dim,
            3,
            padding=1,
            equalized=True,
            initBiasToZero=True
        )
        self.ada1 = AdaptiveInstanceNorm(mapping_dim, new_dim)

        self.conv2 = EqualizedConv2d(
            new_dim,
            new_dim,
            3,
            padding=1,
            equalized=True,
            initBiasToZero=True
        )
        self.ada2 = AdaptiveInstanceNorm(mapping_dim, new_dim)
        self.activation = nn.LeakyReLU(leak)

    def forward(self, w, features):
        features = Upscale2d(features)
        features = self.activation(self.conv1(features))
        features = self.ada1(features, w)

        features = self.activation(self.conv2(features))
        features = self.ada2(features, w)
        return features
    
class Color(nn.Module):
    def __init__(self, in_channel, activation=False):
        super().__init__()
        self.conv = EqualizedConv2d(in_channel,
                                    3,
                                    1,
                                    equalized=True,
                                    initBiasToZero=True)
        self.activation = activation

    def forward(self, x):
        out = self.conv(x)
        if self.activation:  out = torch.sigmoid(out)
        return out
    
class Decoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, mapping_dim=256, leak=0.2):
        super(Decoder, self).__init__()

        depth_list = [256, 256, 128, 64]
        depth_scales = [hidden_dim]
        self.scale_layers = nn.ModuleList()
        self.rgb_layers = nn.ModuleList()
        self.rgb_layers.append(Color(hidden_dim))
        for depth in depth_list:
            last_dim = depth_scales[-1]

            scale_layer = Scaler(last_dim, depth, mapping_dim=mapping_dim, leak=leak)
            self.scale_layers.append(scale_layer)

            self.rgb_layers.append(Color(depth))

            depth_scales.append(depth)

        self.conv0 = EqualizedConv2d(
            input_dim,
            hidden_dim,
            3,
            equalized=True,
            initBiasToZero=True,
            padding=1
        )
        self.ada0 = AdaptiveInstanceNorm(mapping_dim, input_dim)
        self.noise0 = UnitConv()
        self.ada1 = AdaptiveInstanceNorm(mapping_dim, hidden_dim)
        self.noise1 = UnitConv()
        self.activation = nn.LeakyReLU(leak)

        # Retrieved from: https://github.com/Xuanmeng-Zhang/MVCGAN/blob/7791afc36ec3c257120e582470962b513f8a77dc/generators/refinegan.py
        self.blur = Blur()
        self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2,
                            mode='bilinear', 
                            align_corners=False), 
                self.blur)

    def forward(self, w, features, output_size, alpha):
        B = w.size(0)
        img_size = features.size(2)

        features = self.activation(features)
        features = self.ada0(features, w)
        features = self.conv0(features)

        features = self.activation(features)
        features = self.ada1(features, w)

        n_depth = int(np.log2(output_size)) - int(np.log2(img_size))

        for n, layer in enumerate(self.scale_layers):
            if n + 1 > n_depth: break 

            skip = self.rgb_layers[n](features)
            features = layer(w, features)

        assert n_depth >= 0, f"n_depth shouldn't be negative. came across n_depth={n_depth}"
        rgb = self.rgb_layers[n_depth](features)

        if n_depth > 0:
            rgb = alpha * rgb + (1 - alpha) * self.upsample(skip)

        return rgb