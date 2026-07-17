import numpy as np
import torch
from torch import nn
from models.decoder_octave import DecoderUnetOctave 


class OCTENCODER_DECODER(nn.Module):
    """HED network."""

    def __init__(self, device, alpha = 0):
        super(OCTENCODER_DECODER, self).__init__()

        self.device = device
        # Layers.
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.relu = nn.ReLU()
        # Note: ceil_mode – when True, will use ceil instead of floor to compute the output shape.
        #       The reason to use ceil mode here is that later we need to upsample the feature maps and crop the results
        #       in order to have the same shape as the original image. If ceil mode is not used, the up-sampled feature
        #       maps will possibly be smaller than the original images.
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        
        self.decoderMEAN = DecoderUnetOctave(alpha = alpha).to(device)
        self.decoderSTD = DecoderUnetOctave(alpha = alpha).to(device)

    def forward(self, x : torch.tensor):
        # VGG-16 network.
        if x.dim() < 4:
            diff = 4 - x.dim()
            
            x.unsqueeze_(0)
            

        image_h, image_w = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))  # Side output 1.
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))  # Side output 2.
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))  # Side output 3.
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))  # Side output 4.
        pool4 = self.maxpool(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))  # Side output 5.

        features = [conv1_2, conv2_2, conv3_3, conv4_3, conv5_3]
        mean = self.decoderMEAN(features)
        mean = crop(mean, image_h, image_w, 0, 0)
        std = self.decoderSTD(features)
        std = crop(std, image_h, image_w, 0, 0)
        std = nn.Softplus()(std)
        
        

        
        return mean, std

def crop(data1, h, w, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert h <= h1 and w <= w1
    data = data1[:, :, crop_h : crop_h + h, crop_w : crop_w + w]
    return data


if __name__ == '__main__':
    im = torch.rand(10, 3, 500, 500)
    model = OCTENCODER_DECODER('cpu')
    print(im.shape)
    result = model(im)
    print(result.shape)
