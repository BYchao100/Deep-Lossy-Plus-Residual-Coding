import torch
import torch.nn as nn

from compressai.layers import conv3x3
from win_attention import WinBasedAttention


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def subpel_conv1x1(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1), nn.PixelShuffle(r)
    )


def downsample_conv1x1(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for down-sampling."""
    return nn.Sequential(
        nn.PixelUnshuffle(r), nn.Conv2d(in_ch * r ** 2, out_ch, kernel_size=1), 
    )


def maskedconv7x7_parallel(in_ch: int, out_ch: int, mask_type="5P") -> nn.Module:

    if mask_type not in ("5P", "4P", "3P", "2P", "P"):
        raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

    maskedconv = MaskedConv2d("A", in_ch, out_ch, kernel_size=7, padding=3)

    if mask_type=="5P":
        pass
    elif mask_type=="4P":
        maskedconv.mask[:, :, 2, 6] = 0
    elif mask_type == "3P":
        maskedconv.mask[:, :, 2, 5:7] = 0
    elif mask_type == "2P":
        maskedconv.mask[:, :, :, :] = 0
        for i in range(7):
            maskedconv.mask[:, :, i, :(6-i)] = 1
    elif mask_type == "P":
        maskedconv.mask[:, :, :, :] = 0
        maskedconv.mask[:, :, :, :3] = 1

    return maskedconv


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type = "A", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, h // 2 + 1 :, :] = 0

    def forward(self, x):
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)


class SWin_Attention(nn.Module):
    """Shift Window-based multi-head self-attention module."""

    def __init__(self, dim, num_heads=8, window_size=8):
        super().__init__()
        N = dim

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.GELU(),
                    conv3x3(N // 2, N // 2),
                    nn.GELU(),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.GELU()

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            WinBasedAttention(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0),
            conv1x1(N, N),
            WinBasedAttention(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=window_size//2),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class Conv2d_cond(nn.Conv2d):
    def __init__(self, cond_num, *args, **kwargs):
        super().__init__(*args, **kwargs)

        out_ch, in_ch, _, _ = self.weight.size()
        self.weight_cond = nn.Linear(cond_num, out_ch, bias=False)
        self.bias_cond = nn.Linear(cond_num, out_ch, bias=False)
        self.soft_plus = nn.Softplus()

    def forward(self, input, cond):
        out = super().forward(input)
        w_cond = self.soft_plus(self.weight_cond(cond)).unsqueeze(2).unsqueeze(3)
        b_cond = self.bias_cond(cond).unsqueeze(2).unsqueeze(3)

        out = out * w_cond + b_cond

        return out

    def run(self, input, tau):
        out = super().forward(input)
        w_cond = self.soft_plus(self.weight_cond.weight[:, tau]).view(1, -1, 1, 1)
        b_cond = self.bias_cond.weight[:, tau].view(1, -1, 1, 1)

        out = out * w_cond + b_cond

        return out