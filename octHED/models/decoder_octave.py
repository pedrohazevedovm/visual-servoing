import torch
import torch.nn as nn
import torch.nn.functional as F
from models.octave_conv import Conv_BN_ACT

class DoubleConvOctave(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, alpha_in = 0, alpha_mid = None, alpha_out = 0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if not alpha_mid:
            alpha_mid = alpha_out 

        self.double_conv_oct = nn.Sequential(
            Conv_BN_ACT(in_channels,  mid_channels,  kernel_size= 3, alpha_in = alpha_in, alpha_out= alpha_mid, padding = 1, bias = True),
            Conv_BN_ACT(mid_channels, out_channels, kernel_size = 3, alpha_in = alpha_mid,alpha_out= alpha_out, padding = 1, bias = True)
        )

    def forward(self, x):
        return self.double_conv_oct(x)[0] #Return a tuple. But alpha out is 0

class UpOctave(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, alpha = 0.5, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvOctave(in_channels, out_channels, in_channels // 2, alpha_in = 0, alpha_mid = alpha, alpha_out=0)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvOctave(in_channels, out_channels, alpha_in = 0, alpha_mid = alpha, alpha_out=0)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        super().__init__()
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        
        self.head = nn.Sequential(
            upsampling,
            conv2d
        )
    
    def forward(self, x):
        return self.head(x)


class DecoderUnetOctave(nn.Module):
    
    def __init__(self, channels_output = (64, 128, 256, 512, 512), num_classes_decoder = 16, alpha = 0):

        super().__init__()
        assert len(channels_output) == 5
        ch1, ch2, ch3, ch4, ch5 = channels_output
        
        # self.extractor5 = DoubleConvOctave(512, 512)
        # self.up4 = UpOctave(512 + 512, 256)
        # self.up3 = UpOctave(256 + 256,  128)
        # self.up2 = UpOctave(128 + 128,  64)
        # self.up1 = UpOctave(64+64,  64)
        self.extractor5 = DoubleConvOctave(ch5, ch5, alpha_in = 0, alpha_mid = alpha, alpha_out= 0)
        self.up4 = UpOctave(ch5 + ch4, ch3, alpha = alpha)
        self.up3 = UpOctave(2*ch3    , ch2, alpha = alpha)
        self.up2 = UpOctave(2*ch2    , ch1, alpha = alpha)
        self.up1 = UpOctave(2*ch1    , ch1, alpha = alpha)
        
        # 64 -> 16
        self.conv_out = nn.Sequential(
            nn.Conv2d(ch1, num_classes_decoder, kernel_size= 1, padding= 0),
            nn.ReLU()
            )
    
        self.head = SegmentationHead(num_classes_decoder, 1, kernel_size=3)

    def forward(self, features : list):
        assert len(features) == 5
        f1, f2, f3, f4, f5 = features
        x5 = self.extractor5(f5)
        x4 = self.up4(x5, f4)
        x3 = self.up3(x4, f3)
        x2 = self.up2(x3, f2)
        x1 = self.up1(x2, f1)
        
        result = self.conv_out(x1)
        result = self.head(result)
        return result



if __name__ == '__main__':
    batch_size = 10
    t1 = torch.randn(batch_size, 64, 512, 512)   # Nível 1
    t2 = torch.randn(batch_size, 128, 256, 256)  # Nível 2
    t3 = torch.randn(batch_size, 256, 128, 128)  # Nível 3
    t4 = torch.randn(batch_size, 512, 64, 64)    # Nível 4
    t5 = torch.randn(batch_size, 512, 32, 32)    # Bottleneck (Fundo da U)
    chs = (64, 128, 256, 512, 512)
    m = DecoderUnetOctave(chs, num_classes= 16)

    out = m([t1, t2, t3, t4, t5])
    print(out.max())
    print(out.min())