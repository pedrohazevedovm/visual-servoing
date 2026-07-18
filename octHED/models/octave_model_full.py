''' THIS ONE has only a difference with the other octave_model
    - In the vgg flux, there isn't a alpha_in = 0 and alpha_out = 0 block, 
    so the execution is faster than the other one. (Less interporlation)'''

import numpy as np
import torch
from torch import nn

from models.octave_conv import OctaveConv_ACT, OctaveConv


class MaxPoolTuple(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: tuple):
        (high, low) = x
        if low is None:
            return (self.maxpool(high), None)
        elif high is None:
            return (None, self.maxpool(low))
        else:
            return (self.maxpool(high), self.maxpool(low))


class OCTHEDFULL(nn.Module):
    """OCTHEDFULL network."""

    def __init__(self, device, alpha, 
                 octave_layers=['conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2',
                                'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']):
        super(OCTHEDFULL, self).__init__()
        # Layers.
        self.relu = nn.ReLU
        self.maxpool_octave = MaxPoolTuple()

        #The first ones couldn't be octave, because of the padding.
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=35)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
    

        self.conv2_1 = make_maybe_octave(
            'conv2_1',
            octave_layers,
            64,
            128,
            3,
            alpha_in=0,
            alpha_out=alpha,
            padding=1,
            activation_layer=self.relu,
        )

        self.conv2_2 = make_maybe_octave(
            'conv2_2',
            octave_layers,
            128,
            128,
            3,
            alpha_in=alpha,
            alpha_out=alpha,
            padding=1,
            activation_layer=self.relu,
        )

        self.conv3_1 = make_maybe_octave(
            'conv3_1',
            octave_layers,
            128,
            256,
            3,
            alpha_in=alpha,
            alpha_out=alpha,
            padding=1,
            activation_layer=self.relu,
        )

        self.conv3_2 = make_maybe_octave(
            'conv3_2',
            octave_layers,
            256,
            256,
            3,
            alpha_in=alpha,
            alpha_out=alpha,
            padding=1,
            activation_layer=self.relu,
        )

        self.conv3_3 = make_maybe_octave(
            'conv3_3',
            octave_layers,
            256,
            256,
            3,
            alpha_in=alpha,
            alpha_out=alpha,
            padding=1,
            activation_layer=self.relu,
        )

        self.conv4_1 = make_maybe_octave(
            'conv4_1',
            octave_layers,
            256,
            512,
            3,
            alpha_in=alpha,
            alpha_out=alpha,
            padding=1,
            activation_layer=self.relu,
        )

        self.conv4_2 = make_maybe_octave(
            'conv4_2',
            octave_layers,
            512,
            512,
            3,
            alpha_in=alpha,
            alpha_out=alpha,
            padding=1,
            activation_layer=self.relu,
        )

        self.conv4_3 = make_maybe_octave(
            'conv4_3',
            octave_layers,
            512,
            512,
            3,
            alpha_in=alpha,
            alpha_out=alpha,
            padding=1,
            activation_layer=self.relu,
        )

        self.conv5_1 = make_maybe_octave(
            'conv5_1',
            octave_layers,
            512,
            512,
            3,
            alpha_in=alpha,
            alpha_out=alpha,
            padding=1,
            activation_layer=self.relu,
        )

        self.conv5_2 = make_maybe_octave(
            'conv5_2',
            octave_layers,
            512,
            512,
            3,
            alpha_in=alpha,
            alpha_out=alpha,
            padding=1,
            activation_layer=self.relu,
        )

        self.conv5_3 = make_maybe_octave(
            'conv5_3',
            octave_layers,
            512,
            512,
            3,
            alpha_in=alpha,
            alpha_out=alpha,
            padding=1,
            activation_layer=self.relu,
        )

        # Note: ceil_mode – when True, will use ceil instead of floor to compute the output shape.
        #       The reason to use ceil mode here is that later we need to upsample the feature maps and crop the results
        #       in order to have the same shape as the original image. If ceil mode is not used, the up-sampled feature
        #       maps will possibly be smaller than the original images.
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.score_dsn1 = nn.Conv2d(64, 1, 1)  # Out channels: 1.
        self.score_dsn2 = OctaveConv(128, 1, 1, alpha_in = alpha, alpha_out = 0)
        self.score_dsn3 = OctaveConv(256, 1, 1, alpha_in = alpha, alpha_out = 0)
        self.score_dsn4 = OctaveConv(512, 1, 1, alpha_in = alpha, alpha_out = 0)
        self.score_dsn5 = OctaveConv(512, 1, 1, alpha_in = alpha, alpha_out = 0)
        self.score_final = nn.Conv2d(5, 1, 1)

        # Fixed bilinear weights.
        self.weight_deconv2 = make_bilinear_weights(4, 1).to(device)
        self.weight_deconv3 = make_bilinear_weights(8, 1).to(device)
        self.weight_deconv4 = make_bilinear_weights(16, 1).to(device)
        self.weight_deconv5 = make_bilinear_weights(32, 1).to(device)

        # Prepare for aligned crop.
        (
            self.crop1_margin,
            self.crop2_margin,
            self.crop3_margin,
            self.crop4_margin,
            self.crop5_margin,
        ) = self.prepare_aligned_crop()

    # noinspection PyMethodMayBeStatic
    def prepare_aligned_crop(self):
        """Prepare for aligned crop."""
        # Re-implement the logic in deploy.prototxt and
        #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
        # Other reference materials:
        #   hed/include/caffe/layer.hpp
        #   hed/include/caffe/vision_layers.hpp
        #   hed/include/caffe/util/coords.hpp
        #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

        def map_inv(m):
            """Mapping inverse."""
            a, b = m
            return 1 / a, -b / a

        def map_compose(m1, m2):
            """Mapping compose."""
            a1, b1 = m1
            a2, b2 = m2
            return a1 * a2, a1 * b2 + b1

        def deconv_map(kernel_h, stride_h, pad_h):
            """Deconvolution coordinates mapping."""
            return stride_h, (kernel_h - 1) / 2 - pad_h

        def conv_map(kernel_h, stride_h, pad_h):
            """Convolution coordinates mapping."""
            return map_inv(deconv_map(kernel_h, stride_h, pad_h))

        def pool_map(kernel_h, stride_h, pad_h):
            """Pooling coordinates mapping."""
            return conv_map(kernel_h, stride_h, pad_h)

        x_map = (1, 0)
        conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
        conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
        pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

        conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
        conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
        pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

        conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
        conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
        conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
        pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

        conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
        conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
        conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
        pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

        conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
        conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
        conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

        score_dsn1_map = conv1_2_map
        score_dsn2_map = conv2_2_map
        score_dsn3_map = conv3_3_map
        score_dsn4_map = conv4_3_map
        score_dsn5_map = conv5_3_map

        upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
        upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
        upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
        upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

        crop1_margin = int(score_dsn1_map[1])
        crop2_margin = int(upsample2_map[1])
        crop3_margin = int(upsample3_map[1])
        crop4_margin = int(upsample4_map[1])
        crop5_margin = int(upsample5_map[1])

        return (
            crop1_margin,
            crop2_margin,
            crop3_margin,
            crop4_margin,
            crop5_margin,
        )

    def forward(self, x):
        # VGG-16 network.
        image_h, image_w = x.shape[2], x.shape[3]

        conv1_1 = abstract_forward(
            conv=self.conv1_1, x=x, activation=self.relu()
        )
        conv1_2 = abstract_forward(
            conv=self.conv1_2, x=conv1_1, activation=self.relu()
        )  # Side output 1
        pool1 = self.maxpool(conv1_2)

        conv2_1 = abstract_forward(
            conv=self.conv2_1, x=pool1, activation=self.relu()
        )
        conv2_2 = abstract_forward(
            conv=self.conv2_2, x=conv2_1, activation=self.relu()
        )  # Side output 2
        pool2 = self.maxpool_octave(conv2_2)

        conv3_1 = abstract_forward(
            conv=self.conv3_1, x=pool2, activation=self.relu()
        )
        conv3_2 = abstract_forward(
            conv=self.conv3_2, x=conv3_1, activation=self.relu()
        )
        conv3_3 = abstract_forward(
            conv=self.conv3_3, x=conv3_2, activation=self.relu()
        )  # Side output 3
        pool3 = self.maxpool_octave(conv3_3)

        conv4_1 = abstract_forward(
            conv=self.conv4_1, x=pool3, activation=self.relu()
        )
        conv4_2 = abstract_forward(
            conv=self.conv4_2, x=conv4_1, activation=self.relu()
        )
        conv4_3 = abstract_forward(
            conv=self.conv4_3, x=conv4_2, activation=self.relu()
        )  # Side output 4
        pool4 = self.maxpool_octave(conv4_3)

        conv5_1 = abstract_forward(
            conv=self.conv5_1, x=pool4, activation=self.relu()
        )
        conv5_2 = abstract_forward(
            conv=self.conv5_2, x=conv5_1, activation=self.relu()
        )
        conv5_3 = abstract_forward(
            conv=self.conv5_3, x=conv5_2, activation=self.relu()
        )  # Side output 5

        score_dsn1 = self.score_dsn1(conv1_2)
        
        score_dsn2 = self.score_dsn2(conv2_2)[0]
        score_dsn3 = self.score_dsn3(conv3_3)[0]
        score_dsn4 = self.score_dsn4(conv4_3)[0]
        score_dsn5 = self.score_dsn5(conv5_3)[0]

        upsample2 = torch.nn.functional.conv_transpose2d(
            score_dsn2, self.weight_deconv2, stride=2
        )
        upsample3 = torch.nn.functional.conv_transpose2d(
            score_dsn3, self.weight_deconv3, stride=4
        )
        upsample4 = torch.nn.functional.conv_transpose2d(
            score_dsn4, self.weight_deconv4, stride=8
        )
        upsample5 = torch.nn.functional.conv_transpose2d(
            score_dsn5, self.weight_deconv5, stride=16
        )

        # Aligned cropping.
        crop1 = score_dsn1[
            :,
            :,
            self.crop1_margin : self.crop1_margin + image_h,
            self.crop1_margin : self.crop1_margin + image_w,
        ]
        crop2 = upsample2[
            :,
            :,
            self.crop2_margin : self.crop2_margin + image_h,
            self.crop2_margin : self.crop2_margin + image_w,
        ]
        crop3 = upsample3[
            :,
            :,
            self.crop3_margin : self.crop3_margin + image_h,
            self.crop3_margin : self.crop3_margin + image_w,
        ]
        crop4 = upsample4[
            :,
            :,
            self.crop4_margin : self.crop4_margin + image_h,
            self.crop4_margin : self.crop4_margin + image_w,
        ]
        crop5 = upsample5[
            :,
            :,
            self.crop5_margin : self.crop5_margin + image_h,
            self.crop5_margin : self.crop5_margin + image_w,
        ]

        # Concatenate according to channels.
        fuse_cat = torch.cat((crop1, crop2, crop3, crop4, crop5), dim=1)
        fuse = self.score_final(
            fuse_cat
        )  # Shape: [batch_size, 1, image_h, image_w].
        results = [crop1, crop2, crop3, crop4, crop5, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results


def make_bilinear_weights(size, num_channels):
    """Generate bi-linear interpolation weights as up-sampling filters (following FCN paper)."""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (
        1 - abs(og[1] - center) / factor
    )
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False  # Set not trainable.
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def make_maybe_octave(
    layer_name,
    octave_layers,
    in_ch,
    out_ch,
    kernel_size,
    padding,
    alpha_in=0,
    alpha_out=0,
    activation_layer=None,
):
    flag = False
    # Activate a whole layer to octave, avoid bugs. Setup of alpha
    for layer in octave_layers:
        if layer in layer_name:
            flag = True

    if flag == True:
        conv = OctaveConv_ACT(
            in_ch,
            out_ch,
            kernel_size,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            padding=padding,
            activation_layer=activation_layer,
        )
    else:
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
    return conv


def abstract_forward(conv, x, activation=None):
    if isinstance(conv, OctaveConv_ACT):
        y = conv(x)
        if y[1] == None:  # alpha out = 0
            y = y[0]
    elif activation is not None:
        y = activation(conv(x))
    else:
        y = conv(x)

    return y
