import torch
import torch.nn as nn
from models.octave_conv import OctaveConv_ACT





class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels: int):
        """Constructor for SqueezeExcitationBlock.

        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(
            in_channels, in_channels // 4
        )  # divide by 4 is mentioned in the paper, 5.3. Large squeeze-and-excite
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(in_channels // 4, in_channels)
        self.act2 = nn.Hardsigmoid()

    def forward(self, x):
        """Forward pass for SqueezeExcitationBlock."""

        identity = x

        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)

        x = identity * x[:, :, None, None]

        return x
    

class DeepConnection(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, ratio : float, alpha : float, depth : int = 3):
        
        assert depth > 0
        super().__init__()
        padding = kernel_size // 2 
        inter_features = int(in_channels*ratio)
        self.input_conv = OctaveConv_ACT(in_channels, inter_features, kernel_size = 1, alpha_in = 0, alpha_out = alpha)
        feature_extract = []

        for i in range(depth):
            if i < depth - 1:
                feature_extract.append(OctaveConv_ACT(inter_features, inter_features, kernel_size = 3, padding = padding, alpha_in = alpha, alpha_out = alpha))        
            else:
                feature_extract.append(OctaveConv_ACT(inter_features, out_channels, kernel_size= 3, padding = padding, alpha_in = alpha, alpha_out = 0))

        
        self.feature_extract_sequential = nn.Sequential(*feature_extract)
        
        
        self.excitation_block = SqueezeExcitationBlock(out_channels)
    

    def forward(self, x):
        
        x = self.input_conv(x)
        x = self.feature_extract_sequential(x)[0] #Octave sequential
        x = self.excitation_block(x)
        return x

if __name__ == "__main__":
    test = torch.rand((10, 56, 255, 255)) # [B, C, W, H]
    model = DeepConnection(56, 100, ratio = 1, alpha = 0.5, depth= 10)
    
    result = model(test)
    print(model)
    print(result.shape)
        

 