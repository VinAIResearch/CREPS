# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_fc(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    demodulate      = True,     # Apply weight demodulation?
):
    batch_size = x.shape[0]
    out_channels, in_channels = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels]) # [OI]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels) / weight.norm(float('inf'), dim=[1], keepdim=True)) # max_I
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0) # [NOI]
    w = w * styles.unsqueeze(1) # [NOI]

    if demodulate:
        dcoefs = (w.square().sum(dim=[2]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.unsqueeze(2) # [NOI]

    x = torch.einsum('nihw, noi -> nohw', x, w.to(x.dtype)) 
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        weight_norm     = True,     # Apply weight normalization?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / (np.sqrt(in_features) if weight_norm else 1)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not update_emas:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        fc_clamp        = None,         # Clamp the output of fully connected layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.activation = activation
        self.fc_clamp = fc_clamp
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', gain=1):
        assert noise_mode in ['random', 'const', 'none'] # unused
        styles = self.affine(w) 
        x = modulated_fc(x=x, weight=self.weight, styles=styles)

        act_gain = self.act_gain * gain
        act_clamp = self.fc_clamp * gain if self.fc_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d},',
            f'w_dim={self.w_dim:d}, activation={self.activation:s}'])

#----------------------------------------------------------------------------

class SynthesisInput(torch.nn.Module):
    def __init__(self,
            channels,                       # Number of output channels.
            w_dim,                          # Intermediate latent (W) dimensionality.
            resolution,                     # Resolution for input layer.
            rank            = 8,            # Rank of output biline.
            bandwidth       = 10            # Bandwidth of fourier feature.
    ):
        super().__init__()
        self.channels = channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.rank = rank
        self.bandwidth = bandwidth

        freqs = torch.randn([1, channels//2*rank]) * bandwidth
        coords = torch.linspace(0, 1, resolution).reshape(1, -1, 1, 1)
        coords = coords.repeat(1, 1, 2, 1)

        self.register_buffer('freqs', freqs)
        self.register_buffer('coords', coords)
        
        self.weight = torch.nn.Parameter(torch.randn([1, channels, 1, 2*rank])) 
        self.affine = FullyConnectedLayer(w_dim, channels*2*rank, bias_init=1)

    def forward(self, w):
        x = self.coords.repeat(w.shape[0], 1, 1, 1) @ self.freqs
        x = x.reshape(*x.shape[:2], 2, self.channels//2, self.rank)

        x = x.permute(0, 3, 1, 2, 4) * (np.pi * 2)
        x = x.reshape(*x.shape[:3], -1)

        x = torch.cat([x, x+np.pi/2], dim=1) 
        x = torch.sin(x)

        weight = self.weight * self.affine(w).reshape(w.shape[0], self.channels, 1, -1)
        x = x * weight / np.sqrt(self.channels)

        return x

    def extra_repr(self):
        return ' '.join([
            f'channels={self.channels:d}, w_dim={self.w_dim:d}, rank={self.rank}',
            f'resolution={self.resolution:d}, bandwidth={self.bandwidth:0.2f}'
        ])

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToResLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, fc_clamp=None, channels_last=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.fc_clamp = fc_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels)

    def forward(self, x, w):
        styles = self.affine(w) * self.weight_gain
        x = modulated_fc(x=x, weight=self.weight, styles=styles, demodulate=False)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.fc_clamp)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 1 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        is_last,                                # Is this the last block?
        res_channels            = None,         # Number of output residual channels.
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        fc_clamp                = 256,          # Clamp the output of fully connected layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.res_channels = res_channels
        self.is_last = is_last
        self.architecture = architecture if in_channels > 1 else 'orig'
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.num_fc = 0
        self.num_tores = 0

        if in_channels == 1:
            self.input =  SynthesisInput(out_channels, w_dim=w_dim, resolution=resolution)
            self.fc1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim,
                fc_clamp=fc_clamp, channels_last=self.channels_last, **layer_kwargs)

        if in_channels != 1:
            self.fc0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim,
                    fc_clamp=fc_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_fc += 1
            self.fc1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim,
                fc_clamp=fc_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_fc += 1

        if is_last or architecture == 'skip':
            self.tores = ToResLayer(out_channels, self.res_channels, w_dim=w_dim,
                fc_clamp=fc_clamp, channels_last=self.channels_last)
            self.num_tores += 1

    def forward(self, x, res, ws, force_fp32=False, update_emas=False, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, (self.in_channels == 1) + self.num_fc + self.num_tores, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if self.in_channels == 1:
            x = self.input(next(w_iter)).to(dtype=dtype, memory_format=memory_format)
        else:
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 1:
            x = self.fc1(x, next(w_iter), **layer_kwargs)
        else:
            x = self.fc0(x, next(w_iter), **layer_kwargs)
            x = self.fc1(x, next(w_iter), **layer_kwargs)

        # ToRes.
        if self.is_last or self.architecture == 'skip':
            y = self.tores(x, next(w_iter))
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            res = res.add(y) if res is not None else y

        assert x.dtype == dtype
        assert res is None or res.dtype == torch.float32
        return x, res

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        biline_channels = 32,       # Number of biline channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        assert channel_base > 4 and channel_max > 1 and biline_channels > img_channels
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.biline_channels = biline_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 1
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 1
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=img_resolution, 
                    res_channels=biline_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_fc
            if is_last:
                self.num_ws += block.num_tores
            setattr(self, f'b{res}', block)

        for res in self.block_resolutions[1:]:
            decoder = torch.nn.Sequential(
                FullyConnectedLayer(biline_channels, biline_channels*2, activation='lrelu'),
                FullyConnectedLayer(biline_channels*2, biline_channels*4, activation='lrelu'),
                FullyConnectedLayer(biline_channels*4, biline_channels*2, activation='lrelu'),
                FullyConnectedLayer(biline_channels*2, biline_channels, activation='lrelu')
            )
            setattr(self, f'decoder_{res}', decoder)
        self.decoder_last = torch.nn.Sequential(
            FullyConnectedLayer(biline_channels, biline_channels*2, activation='lrelu'),
            FullyConnectedLayer(biline_channels*2, biline_channels*4, activation='lrelu'),
            FullyConnectedLayer(biline_channels*4, biline_channels*2, activation='lrelu'),
            FullyConnectedLayer(biline_channels*2, biline_channels, activation='lrelu')
        )
        self.decoder_img = FullyConnectedLayer(biline_channels, img_channels)

        self.block0 = SynthesisBlock(biline_channels, biline_channels*4, w_dim=w_dim, resolution=img_resolution,
                res_channels=img_channels, is_last=False, use_fp16=True, fc_clamp=256)
        self.block1 = SynthesisBlock(biline_channels*4, biline_channels*2, w_dim=w_dim, resolution=img_resolution,
                res_channels=img_channels, is_last=True, use_fp16=True, fc_clamp=256)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, (res == 4) + block.num_fc + block.num_tores))
                w_idx += block.num_fc

        x = biline = None
        feature = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, biline = block(x, biline, cur_ws, **block_kwargs)
            if biline is not None:
               x_line, y_line = (biline * np.sqrt(0.5)).chunk(2, dim=-1)
               new_f = x_line @ y_line.transpose(-2, -1)
               batch, channel, height, width = new_f.shape
               new_f = new_f.permute(0, 2, 3, 1).reshape(-1, channel)
               if feature is None:
                  feature = new_f
               else:
                  decoder = getattr(self, f'decoder_{res}')
                  feature = decoder(feature) + new_f

        feature = self.decoder_last(feature); img = self.decoder_img(feature)
        feature = feature.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        img = img.reshape(batch, height, width, -1).permute(0, 3, 1, 2)

        ws = ws[:, -1:, :].repeat(1, 3, 1)
        feature, img = self.block0(feature, img, ws, **block_kwargs)
        feature, img = self.block1(feature, img, ws, **block_kwargs)
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs          # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------
