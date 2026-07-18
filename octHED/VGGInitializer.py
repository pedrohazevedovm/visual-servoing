import pickle

import torch
from torch import nn
from torchvision import models

# CUSTOM IMPORTS ####
from models.octave_conv import *


class AbstractVGGInitializer:
    def load(self, net, device):
        raise NotImplementedError


class CaffeVGGInitializer(AbstractVGGInitializer):
    def __init__(self, path, only_vgg=True):
        self.path = path
        self.only_vgg = only_vgg

    def load(self, net, device=None):
        with open(self.path, 'rb') as f:
            pretrained_params = pickle.load(f)

        vgg_layers_name = [
            'conv1_1',
            'conv1_2',
            'conv2_1',
            'conv2_2',
            'conv3_1',
            'conv3_2',
            'conv3_3',
            'conv4_1',
            'conv4_2',
            'conv4_3',
            'conv5_1',
            'conv5_2',
            'conv5_3',
        ]

        for name, param in net.named_parameters():
            _, layer_name, var_name = name.split('.')
            if (not self.only_vgg) or (layer_name in vgg_layers_name):
                param.data.copy_(
                    torch.from_numpy(pretrained_params[layer_name][var_name])
                )


class OctaveVGGInitializer(AbstractVGGInitializer):
    def load(self, net: nn.DataParallel, device: str):

        # A Data Parallel atribute
        net = net.module

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(
            device
        )

        vgg_layers = [m for m in vgg.features if isinstance(m, nn.Conv2d)]

        hed_blocks = [
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv1' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv2' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv3' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv4' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv5' in name
            ],
        ]

        vgg_blocks = [
            vgg_layers[:2],
            vgg_layers[2:4],
            vgg_layers[4:7],
            vgg_layers[7:10],
            vgg_layers[10:13],
        ]

        

        for hed_set, vgg_set in zip(hed_blocks, vgg_blocks):
            # assert len(hed_set) == len(vgg_blocks), f"{len(hed_set)} and {len(vgg_blocks)}"
            for hed_layer, vgg_layer in zip(hed_set, vgg_set):
                if isinstance(hed_layer, nn.Conv2d):
                    hed_layer.weight.data.copy_(vgg_layer.weight.data)
                    hed_layer.bias.data.copy_(vgg_layer.bias.data)
                else:
                    self._transfer_octave(hed_layer, vgg_layer)

    def _transfer_octave(self, oct_layer, vgg_conv):
        oc = oct_layer.conv

        if oc.conv_h2h is not None:
            oc.conv_h2h.weight.data.copy_(
                vgg_conv.weight.data[
                    : oc.conv_h2h.out_channels, : oc.conv_h2h.in_channels
                ]
            )
            oc.conv_h2h.bias.data.copy_(
                vgg_conv.bias.data[: oc.conv_h2h.out_channels]
            )

        if oc.conv_l2l is not None:
            nn.init.xavier_normal_(oc.conv_l2l.weight)
            nn.init.zeros_(oc.conv_l2l.bias)

        if oc.conv_h2l is not None:
            nn.init.xavier_normal_(oc.conv_h2l.weight)
            nn.init.zeros_(oc.conv_h2l.bias)

        if oc.conv_l2h is not None:
            nn.init.xavier_normal_(oc.conv_l2h.weight)
            nn.init.zeros_(oc.conv_l2h.bias)


class FourierVGGInitializer(AbstractVGGInitializer):
    def load():
        pass

class ExitVGGInitializer(AbstractVGGInitializer):
    def load(self, net: nn.DataParallel, device: str):
        # A Data Parallel atribute
        net = net.module

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(
            device
        )

        vgg_layers = [m for m in vgg.features if isinstance(m, nn.Conv2d)]

        hed_blocks = [
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv1' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv2' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv3' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv4' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv5' in name
            ],
        ]

        exitation_blocks = [
            layer 
            for (name, layer) in net.named_children()
            if 'exitation' in name
        ]

        vgg_blocks = [
            vgg_layers[:2],
            vgg_layers[2:4],
            vgg_layers[4:7],
            vgg_layers[7:10],
            vgg_layers[10:13],
        ]

        for hed_set, vgg_set in zip(hed_blocks, vgg_blocks):
            # assert len(hed_set) == len(vgg_blocks), f"{len(hed_set)} and {len(vgg_blocks)}"
            for hed_layer, vgg_layer in zip(hed_set, vgg_set):
                if isinstance(hed_layer, nn.Conv2d):
                    hed_layer.weight.data.copy_(vgg_layer.weight.data)
                    hed_layer.bias.data.copy_(vgg_layer.bias.data)

        for layer_exitation in exitation_blocks:
            nn.init.xavier_normal_(layer_exitation.linear1.weight)
            nn.init.constant_(layer_exitation.linear1.bias, 0)
            nn.init.xavier_normal_(layer_exitation.linear2.weight)
            nn.init.constant_(layer_exitation.linear2.bias, 1)

class EncDecInitializer(AbstractVGGInitializer):
    def load(self, net: nn.DataParallel, device: str):

        # A Data Parallel atribute
        net = net.module

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)

        vgg_layers = [m for m in vgg.features if isinstance(m, nn.Conv2d)]

        hed_blocks = [
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv1' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv2' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv3' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv4' in name
            ],
            [
                layer
                for (name, layer) in net.named_children()
                if 'conv5' in name
            ],
        ]

        vgg_blocks = [
            vgg_layers[:2],
            vgg_layers[2:4],
            vgg_layers[4:7],
            vgg_layers[7:10],
            vgg_layers[10:13],
        ]

        decoder_blocks = [
            #Two layers, MEAN and STD
            layer for (name, layer) in net.named_children() if 'decoder' in name
        ]

        for hed_set, vgg_set in zip(hed_blocks, vgg_blocks):
            # assert len(hed_set) == len(vgg_blocks), f"{len(hed_set)} and {len(vgg_blocks)}"
            for hed_layer, vgg_layer in zip(hed_set, vgg_set):
                if isinstance(hed_layer, nn.Conv2d):
                    hed_layer.weight.data.copy_(vgg_layer.weight.data)
                    hed_layer.bias.data.copy_(vgg_layer.bias.data)
        


        for decoder in decoder_blocks:
            for name, layer in decoder.named_modules():
                
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    

def weights_init(m : torch.tensor):
        """Weight initialization function."""
        if isinstance(m, nn.Conv2d):
            # Initialize: m.weight.
            if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
                # Constant initialization for fusion layer in HED network.
                torch.nn.init.constant_(m.weight, 0.2)
            else:
                # Zero initialization following official repository.
                # Reference: hed/docs/tutorial/layers.md
                m.weight.data.zero_()
            # Initialize: m.bias.
            if m.bias is not None:
                # Zero initialization.
                m.bias.data.zero_()
        elif isinstance(m, OctaveConv):
            if m.conv_h2h is not None:
                nn.init.zeros_(m.conv_h2h.weight)
                nn.init.zeros_(m.conv_h2h.bias)

            if m.conv_l2l is not None:
                nn.init.zeros_(m.conv_l2l.weight)
                nn.init.zeros_(m.conv_l2l.bias)

            if m.conv_h2l is not None:
                nn.init.zeros_(m.conv_h2l.weight)
                nn.init.zeros_(m.conv_h2l.bias)

            if m.conv_l2h is not None:
                nn.init.zeros_(m.conv_l2h.weight)
                nn.init.zeros_(m.conv_l2h.bias)


def default_initializer(net : nn.Module):
    net.apply(weights_init)

def chose_initializer(args, net, device):
    if not args.fine_tuning:
        tag = "NULL"
    elif args.model == 'OCTHED' \
        or args.model == 'OCTHEDFULL' \
        or args.model == 'OCTHEDFULL_SIDE' \
        or args.model == "OCTHEDFULL_SIDE_EXTRA"\
        or args.model == 'EXCITOCTHED':
        initializer = OctaveVGGInitializer()
        tag = "OCTHED"
    elif args.model == 'HED':
        if args.vgg16_caffe:
            initializer = CaffeVGGInitializer(path=args.vgg16_caffe, only_vgg=True)
            tag = "HED CAFFE"
        else:
            initializer = OctaveVGGInitializer()
            tag = "HED"
    elif args.model ==  'FFCHED':
        raise NotImplementedError('Not implemented yet!')
    elif args.model in ['EXCITHED', 'DENSEHED']: # Tupla para múltiplos tipos
        initializer = ExitVGGInitializer()
        tag = "EXCITHED" 
    elif args.model.upper() == 'ENCODER_DECODER':
        initializer = EncDecInitializer()
        tag = "ENCODER_DECODER"
    elif args.model.upper() == 'OCTENCODER_DECODER':
        initializer = EncDecInitializer()
        tag = "OCTENCODER_DECODER"
    else: 
        raise ValueError()
    
    print(f"FINE-TUNING {tag}")
    

    if args.fine_tuning:
        initializer.load(net, device) 