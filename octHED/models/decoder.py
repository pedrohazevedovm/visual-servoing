import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

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


class DecoderUnet(nn.Module):
    
    def __init__(self, channels_output = (64, 128, 256, 512, 512), num_classes_decoder = 8):

        super().__init__()
        assert len(channels_output) == 5
        ch1, ch2, ch3, ch4, ch5 = channels_output

        self.extractor5 = DoubleConv(ch5, ch5)
        self.up4 = Up(ch5 + ch4, ch3)
        self.up3 = Up(2*ch3,  ch2)
        self.up2 = Up(2*ch2,  ch1)
        self.up1 = Up(2*ch1,  ch1)
        
        #64 -> 16
        self.conv_out = nn.Sequential(
            nn.Conv2d(ch1, num_classes_decoder, kernel_size= 1, padding = 0),
            nn.ReLU()
        )

        #16 -> 1 (map)
        self.head = SegmentationHead(num_classes_decoder, 1, kernel_size= 3)


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
    m = DecoderUnet(chs, num_classes= 16)

    out = m([t1, t2, t3, t4, t5])
    print(out.max())
    print(out.min())