from torch import nn, Tensor
from torch.nn import TransformerEncoder
import os
import torch
from model.network import context_aggregator
from model.network import CBAM
import torch.nn.functional as F
import torch.fft as fft

padding_mode = 'reflect'
#reflect, replicate,,zeros
predim = 64

def RGB2MP(imgs):
    fft_imgs = fft.rfftn(imgs, dim=(2,3), norm='ortho')
    r = torch.real(fft_imgs)
    i = torch.imag(fft_imgs)
    return torch.cat([r,i], dim=1)


class WaveTransform(nn.Module):
    """Layer to perform wave transform
    """
    def __init__(self, in_channel, out_channel, h, w):
        super(WaveTransform, self).__init__()
        #self.weights = torch.nn.Parameter(torch.rand(1, 1, h, w) * 0.02)
        self.multiheads = nn.Conv2d(in_channel, out_channel, 1, bias=False, groups=1)

    def forward(self, x):
        x = self.multiheads(x)
        return x

class Act(nn.Module):
    def __init__(self):
        super(Act, self).__init__()
        self.act = nn.SiLU(True)

    def forward(self, x):
        return self.act(x)
class WaveTransformBlock(nn.Module):
    """Layer to perform wave transform
    """

    def __init__(self, in_channel, out_channel, h, w):
        super(WaveTransformBlock, self).__init__()
        self.wavet = WaveTransform(in_channel, out_channel, h, w)
        self.act = Act()
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.wavet(x)
        x = self.act(x)
        x = self.bn(x)
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2, stride=stride, bias=False,
                      padding_mode=padding_mode),
            Act(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.net(x)
def MP2Disp(MP_map):
    _, c, _, _ = MP_map.shape
    r = MP_map[:,0:c//2,:,:]
    i = MP_map[:,c//2:,:,:]
    MP_map_complex = r + (1j*i)
    rimg = fft.irfftn(MP_map_complex, dim=(2,3), norm='ortho')
    return rimg
def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

class FLBlock(nn.Module):
    def __init__(self, in_channels, out_channels, h, w):
        super().__init__()

        self.net_global = nn.Sequential(
            WaveTransformBlock(2*in_channels, out_channels, h, w),
            WaveTransformBlock(out_channels, 2*out_channels, h, w),
        )

        self.net_local = nn.Sequential(
            CNNBlock(in_channels, out_channels//2),
            CNNBlock(out_channels//2, out_channels),
        )
        self.compression = nn.Sequential(
            CNNBlock(2*out_channels, out_channels,stride=2),
        )

    def forward(self, x):
        b, _, h, w = x.shape
        img = x

        x = RGB2MP(x)
        x = self.net_global(x)
        x = MP2Disp(x)
        if x.size(2) != h or x.size(3) != w:
            x = F.interpolate(x, [h, w], mode="bilinear", align_corners=False)

        img = self.net_local(img)

        x = self.compression(torch.cat([x, img], dim=1))
        # print(x.shape)

        return x
class FLBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, h, w):
        super().__init__()

        self.net_global = nn.Sequential(
            WaveTransformBlock(2 * in_channels, out_channels, h, w),
            WaveTransformBlock(out_channels, 2 * out_channels, h, w),
        )

        self.net_local = nn.Sequential(
            CNNBlock(in_channels, out_channels // 2),
            CNNBlock(out_channels // 2, out_channels),
        )
        self.compression = nn.Sequential(
            CNNBlock(2*out_channels, out_channels),
        )

    def forward(self, x):
        b, _, h, w = x.shape
        img = x

        x = RGB2MP(x)
        x = self.net_global(x)
        x = MP2Disp(x)
        if x.size(2) != h or x.size(3) != w:
            x = F.interpolate(x, [h, w], mode="bilinear", align_corners=False)

        img = self.net_local(img)

        x = self.compression(torch.cat([x, img], dim=1))
        # print(x.shape)

        return x

class FLBCDnet(nn.Module):
    def __init__(self, in_channels, out_channels, h=64, w=64):
        super().__init__()

        self.flblock1 = FLBlock(in_channels=in_channels//2, out_channels=out_channels//8, h=h, w=w)
        self.flblock2 = FLBlock(in_channels=in_channels//2, out_channels=out_channels//8, h=h, w=w)
        self.flblock3 = FLBlock2(in_channels=out_channels//4, out_channels=out_channels, h=h//2, w=w//2)

    def forward(self, x):
        b, _, h, w = x.shape
        x1 = x[:,0:24,:,:]
        x2 = x[:,24:48,:,:]
        x1 = self.flblock1(x1)
        x2 = self.flblock2(x2)
        # print(x1.shape)
        xdif = x1-x2
        x = torch.cat([x1,x2], dim=1)
        x = self.flblock3(x)
        # print(x.shape)
        return [xdif,x]


class SLnet(nn.Module):
    def __init__(self,in_channels=48,out_c1 =160,out_channels=1280,h=64,w = 64):
        super().__init__()
        self.CNN1 = nn.Sequential(
            CNNBlock(in_channels//2, out_channels//4),
            CNNBlock(out_channels//4, out_c1,stride=2)
        )
        self.CNN2 = nn.Sequential(
            CNNBlock(out_c1*2, out_channels//2),
            CNNBlock(out_channels//2, out_channels)
        )
    def forward(self, x):

        x1 = x[:, 0:24, :, :]
        x2 = x[:, 24:48, :, :]
        x1 = self.CNN1(x1)
        x2 = self.CNN1(x2)
        xdif = x1 - x2
        x = torch.cat([x1, x2], dim=1)
        x = self.CNN2(x)
        return [xdif, x]


if __name__ == "__main__":
    x1 = torch.randn(4,48,64,64).cuda()
    net = FLBCDnet(in_channels=48,out_channels=1280).cuda()
    x1,x = net(x1)
    print(x1.shape,x.shape)



