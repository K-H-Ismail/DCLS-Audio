# Copyright 2023 by Ismail Khalfaoui-Hassani, ANITI Toulouse.
#
# All rights reserved.
#
# This file is part of the Dcls-Audio package, and
# is released under the "MIT License Agreement".
# Please see the LICENSE file that should have been included as part
# of this package.

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from DCLS.construct.modules import Dcls2d
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

AUDIOSET_DEFAULT_MEAN = (-18.2696,)
AUDIOSET_DEFAULT_STD = (30.5735,)

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 527,
        "input_size": (1, 128, 1001),
        "pool_size": None,
        "crop_pct": 1.0,
        "interpolation": "bicubic",
        "mean": AUDIOSET_DEFAULT_MEAN,
        "std": AUDIOSET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }

default_cfgs = {
    "convnext_audio_tiny": _cfg()
}

class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        **kwargs
    ):
        super().__init__()
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=(4, 4), stride=(4, 4)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, Dcls2d)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
    "convnext_dcls_gauss_tiny_1k": "https://zenodo.org/record/8029747/files/convnext_dcls_gauss_tiny_1k_224_ema.pth",
    "convnext_tiny_1k_audio": "https://zenodo.org/record/8020843/files/convnext_tiny_471mAP.pth",
}

# Helper function that replaces all ".int." patterns
# by "[int]" in a character string
def replace_dots_brackets(name):
    name_split = name.split(".")
    name_split = ["[" + i + "]" if i.isdigit() else "." + i for i in name_split]
    return "".join(name_split[:-1]), name_split[-1][1:]


# Helper function that replaces all the
# 2D depthwise separable convolution in
# a model by synchronized Dcls2d ones
def replace_depthwise_dcls(
    model, dilated_kernel_size=23, kernel_count=26, version="gauss"
):
    in_channels, P, SIG = 0, None, None
    # Loop over all model modules
    for name, module in model.named_modules():
        # if the module is a depthwise separable Conv2d module
        if (
            isinstance(module, nn.Conv2d)
            and module.groups == module.in_channels == module.out_channels
        ):
            name_eval, last_layer = replace_dots_brackets(name)
            dcls_conv = Dcls2d(
                module.in_channels,
                module.out_channels,
                kernel_count=kernel_count,
                dilated_kernel_size=dilated_kernel_size,
                padding=dilated_kernel_size // 2,
                groups=module.in_channels,
                version=version,
                bias=module.bias is not None,
            )
            nn.init.normal_(dcls_conv.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(dcls_conv.bias, 0)

            # Synchronise positions and standard
            # deviations belonging to the same stage
            if in_channels < module.in_channels:
                in_channels = module.in_channels
                P, SIG = dcls_conv.P, dcls_conv.SIG

            dcls_conv.P, dcls_conv.SIG = P, SIG

            setattr(eval("model" + name_eval), last_layer, dcls_conv)
    return model


@register_model
def convnext_audio_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    model.default_cfg = default_cfgs["convnext_audio_tiny"]    
    if pretrained:
        url = (
            model_urls["convnext_tiny_22k"]
            if in_22k
            else model_urls["convnext_tiny_1k"]
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)

    stem_audio = nn.Conv2d(1, 96, kernel_size=(2, 16), stride=(2, 16))
    trunc_normal_(stem_audio.weight, std=0.02)
    nn.init.constant_(stem_audio.bias, 0)
    model.downsample_layers[0][0] = stem_audio

    return model


@register_model
def convnext_dcls_audio_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    model.default_cfg = default_cfgs["convnext_audio_tiny"]
    # Replace all the 2D depthwise separable convolutions
    # in the model by synchronized Dcls2d ones.
    model = replace_depthwise_dcls(
        copy.deepcopy(model), dilated_kernel_size=23, kernel_count=26, version="gauss"
    )
    if pretrained:
        url = model_urls["convnext_dcls_gauss_tiny_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    stem_audio = nn.Conv2d(1, 96, kernel_size=(2, 16), stride=(2, 16))
    trunc_normal_(stem_audio.weight, std=0.02)
    nn.init.constant_(stem_audio.bias, 0)
    model.downsample_layers[0][0] = stem_audio
    return model


@register_model
def convnext_dcls_audio_tiny17(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    model.default_cfg = default_cfgs["convnext_audio_tiny"]
    # Replace all the 2D depthwise separable convolutions
    # in the model by synchronized Dcls2d ones.
    model = replace_depthwise_dcls(
        copy.deepcopy(model), dilated_kernel_size=17, kernel_count=34, version="v1"
    )
    if pretrained:
        url = model_urls["convnext_dcls_v1_tiny_1k"]
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    stem_audio = nn.Conv2d(1, 96, kernel_size=(2, 16), stride=(2, 16))
    trunc_normal_(stem_audio.weight, std=0.02)
    nn.init.constant_(stem_audio.bias, 0)
    model.downsample_layers[0][0] = stem_audio
    return model


