from collections import OrderedDict
import torch
from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models.utils import load_state_dict_from_url
# from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from model.network import CBAM,ChannelAttention
from baseresnet import resnet50, resnet101
from encoder_rgb import encoderres
from collections import OrderedDict
from typing import Optional, Dict

from torch import nn, Tensor
from torch.nn import functional as F

class SmallDeepLab2(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.cbam = CBAM(in_planes=2048+512)
        self.cam1 = ChannelAttention(in_planes=512)
        self.cam2 = ChannelAttention(in_planes=2048*2)
        self.conv = nn.Sequential(
            nn.Conv2d(2048*2, 2048, kernel_size=3, padding=3 // 2, stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(2048)
        )
        self.conv_out = nn.Conv2d(2560,2,3)
    def forward(self, x1: Tensor, x2: Tensor) -> Dict[str, Tensor]:
        input_shape = x1.shape[-2:]
        features = self.backbone(x1,x2)
        x1r2 = features["x1_res2"]
        x2r2 = features["x2_res2"]
        xres2 = x1r2-x2r2
        x1r4 = features["x1_res4"]
        x2r4 = features["x2_res4"]
        xres4 = torch.cat([x1r4,x2r4],1)
        xres4 = self.conv(xres4)
        xdif = torch.cat([xres4,xres2],1)
        xdif = self.cbam(xdif)
        resultrgb = OrderedDict()
        y = self.classifier(xdif)
        y = F.interpolate(y, size=input_shape, mode='bilinear', align_corners=False)
        resultrgb.update({"xres2":xres2,"xres4":xres4,"xdif":xdif,"out":y})
        return resultrgb


def decoder_rgb(pretrained=True, resnet="res50", head_in_ch=2560, num_classes=2):
    net = SmallDeepLab2(
        backbone=encoderres(pretrained=pretrained,resnet=resnet),
        classifier=DeepLabHead(head_in_ch, num_classes)
    )
    return net

if __name__ == "__main__":
    pass