import math

from torch import nn
import torch.nn.functional as F 

class OctaveConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha_in=0.5,
        alpha_out=0.5,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        assert stride == 1 or stride == 2, 'Stride should be 1 or 2'
        self.stride = stride

        self.is_dw = groups == in_channels

        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, (
            'Alpha should be in the interval from 0 to 1'
        )
        self.alpha_in, self.alpha_out = alpha_in, alpha_out

        # print("alpha_in: ", alpha_in, " alpha_out: ", alpha_out)
        self.conv_l2l = (
            None
            if alpha_in == 0 or alpha_out == 0
            else nn.Conv2d(
                in_channels=int(alpha_in * in_channels),
                out_channels=int(alpha_out * out_channels),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=math.ceil(alpha_in * groups),
                bias=bias,
            )
        )

        self.conv_l2h = (
            None
            if alpha_in == 0 or alpha_out == 1 or self.is_dw
            else nn.Conv2d(
                in_channels=int(alpha_in * in_channels),
                out_channels=out_channels - int(alpha_out * out_channels),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )

        self.conv_h2l = (
            None
            if alpha_in == 1 or alpha_out == 0 or self.is_dw
            else nn.Conv2d(
                in_channels=in_channels - int(alpha_in * in_channels),
                out_channels=int(alpha_out * out_channels),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )

        self.conv_h2h = (
            None
            if alpha_in == 1 or alpha_out == 1
            else nn.Conv2d(
                in_channels=in_channels - int(alpha_in * in_channels),
                out_channels=out_channels - int(alpha_out * out_channels),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=math.ceil(groups - alpha_in * groups),
                bias=bias,
            )
        )

    def forward(self, x):

        x_h, x_l = x if type(x) is tuple else (x, None)

        # fig, axs = plt.subplots

        x_h = self.downsample(x_h) if self.stride == 2 else x_h

        x_h2h = self.conv_h2h(x_h) if (self.conv_h2h is not None) else None
        x_h2l = (
            self.conv_h2l(self.downsample(x_h))
            if self.alpha_out > 0 and not self.is_dw
            else None
        )

        if x_l is not None:
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None

            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                # print("Testes ", x_l2h.size(), x_h2h.size())
                if x_l2h.size()[-1] != x_h2h.size()[-1]:
                    x_l2h = nn.functional.pad(
                        x_l2h, (0, 1), mode='constant', value=0
                    )
                if x_l2h.size()[-2] != x_h2h.size()[-2]:
                    x_l2h = nn.functional.pad(
                        x_l2h, (0, 0, 0, 1), mode='constant', value=0
                    )

                x_h = x_l2h + x_h2h
                x_l = (
                    x_h2l + x_l2l
                    if x_h2l is not None and x_l2l is not None
                    else None
                )

                return x_h, x_l

        else:
            return x_h2h, x_h2l


class Conv_BN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha_in=0.5,
        alpha_out=0.5,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        norm_layer=nn.BatchNorm2d,
    ):
        print('CONV_BN')
        super(Conv_BN, self).__init__()
        self.conv = OctaveConv(
            in_channels,
            out_channels,
            kernel_size,
            alpha_in,
            alpha_out,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.bn_h = (
            None
            if alpha_out == 1
            else norm_layer(int(out_channels * (1 - alpha_out)))
        )
        self.bn_l = (
            None
            if alpha_out == 0
            else norm_layer(int(out_channels * alpha_out))
        )

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l


class OctaveConv_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha_in=0.5,
        alpha_out=0.5,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation_layer=nn.ReLU,
    ):
        super(OctaveConv_ACT, self).__init__()
        self.conv = OctaveConv(
            in_channels,
            out_channels,
            kernel_size,
            alpha_in,
            alpha_out,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(x_h) if x_h is not None else None
        x_l = self.act(x_l) if x_l is not None else None
        return x_h, x_l


class Conv_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha_in=0.5,
        alpha_out=0.5,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
    ):
        super(Conv_BN_ACT, self).__init__()
        print('CONV_BN_ACT')
        self.conv = OctaveConv(
            in_channels,
            out_channels,
            kernel_size,
            alpha_in,
            alpha_out,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.bn_h = (
            None
            if alpha_out == 1
            else norm_layer(int(out_channels * (1 - alpha_out)))
        )
        self.bn_l = (
            None
            if alpha_out == 0
            else norm_layer(int(out_channels * alpha_out))
        )
        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l


class TransposeOctConv(nn.Module):

    """This is the implementation of Octave Transpose Conv from paper https://arxiv.org/abs/1906.12193"""

    def __init__(
        self,
        in_chn,
        out_chn,
        alpha_in,
        alpha_out,
        kernel=2,
        ):

        super(TransposeOctConv, self).__init__()

        (self.alpha_in, self.alpha_out) = alpha_in, alpha_out

        assert 1 > self.alpha_in >= 0 and 1 > self.alpha_out >= 0, \
            'alphas values must be bound between 0 and 1, it could be 0 but not 1'

        self.htoh = nn.ConvTranspose2d(in_chn - int(self.alpha_in
                * in_chn), out_chn - int(self.alpha_out * out_chn),
                kernel, 2)
        self.htol = (nn.ConvTranspose2d(in_chn - int(self.alpha_in
                     * in_chn), int(self.alpha_out * out_chn), kernel,
                     2) if self.alpha_out > 0 else None)
        self.ltol = (nn.ConvTranspose2d(int(self.alpha_in * in_chn),
                     int(self.alpha_out * out_chn), kernel,
                     2) if self.alpha_out > 0 and self.alpha_in
                     > 0 else None)
        self.ltoh = (nn.ConvTranspose2d(int(self.alpha_in * in_chn),
                     out_chn - int(self.alpha_out * out_chn), kernel,
                     2) if self.alpha_in > 0 else None)

    def forward(self, x) -> tuple:
        (high, low) = (x if isinstance(x, tuple) else (x, None))

        if self.htoh is not None:
            htoh = self.htoh(high)
        if self.htol is not None:
            htol = self.htol(F.avg_pool2d(high, 2, 2))
        if self.ltol is not None and low is not None:
            ltol = self.ltol(low)
        if self.ltoh is not None and low is not None:
            ltoh = F.interpolate(self.ltoh(low), scale_factor=2,
                                 mode='nearest')

        # it will behave as normal Transpose Conv operation

        if self.alpha_in == 0 and self.alpha_out == 0:
            return (htoh, None)

        # case where we don't want a low frequency map as output

        if self.alpha_out == 0:
            return (htoh.add_(ltoh), None)

        # otherwise add feature maps and return both high and low freq maps

        htoh.add_(ltoh)
        ltol.add_(htol)

        return (htoh, ltol)