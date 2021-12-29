import torch
import torch.nn as nn
#import torch.nn.functional as F


class UNet(nn.Module):
    """Instantiate a simple UNet model"""

    def __init__(self, config):
        
        super(UNet, self).__init__()
        self.num_classes = config['num_classes']
        self.inc = 3
        self.outc = config['num_filters']
        self.img_h = config['image_dim'][0]
        self.img_w = config['image_dim'][1]
        
        self.down0 = RepeatConv(self.inc, self.outc)
        self.down1 = DownConv(self.outc, self.outc*2)
        self.down2 = DownConv(self.outc*2, self.outc*4)
        self.down3 = DownConv(self.outc*4, self.outc*8)
        self.base = DownConv(self.outc*8, self.outc*16)
        self.up3 = UpConv(self.outc*16, self.outc*8)
        self.up2 = UpConv(self.outc*8, self.outc*4)
        self.up1 = UpConv(self.outc*4, self.outc*2)
        self.up0 = UpConv(self.outc*2, self.num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x (array): BxCxHxW, input tensor.
        Returns:
            logits: BxDxHxW, where D represents number of classes
        """

        # Define the forward  pass and get the logits for classification.
        x1 = self.down0(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y4 = self.base(x4)
        y3 = self.up3(y4, x4)
        y2 = self.up2(y3, x3)
        y1 = self.up1(y2, x2)
        logits = self.up0(y1, x1)

        return logits

class RepeatConv(nn.Module):
    """Two convolutions with ReLU activation in succession"""
    
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.repeat_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
            )
        self.bn = nn.BatchNorm2d(out_channels)
            
    def forward(self, x):
        x = self.repeat_conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return x

class DownConv(nn.Module):
    """RepeatConv followed by downsampling"""
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.repeat_conv = RepeatConv(in_channels, out_channels, batch_norm)
        self.downsample = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.repeat_conv(x)
        x = self.downsample(x)
        return x

class UpConv(nn.Module):
    """First upConv, then concatenate with skip connection"""
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels//2, 
                                         kernel_size=2, stride=2)
        self.repeat_conv = RepeatConv(in_channels, out_channels, batch_norm)
    
    def forward(self, x1, x2):
        """x1 is the lower layer, so to speak"""
        x1 = self.upconv(x1)
        try:
            x = torch.cat([x2, x1], dim=1)
        except:
            print(x1.size())
            print(x2.size())
        x = self.repeat_conv(x)
        return x
        
        
        