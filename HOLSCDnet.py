from collections import OrderedDict
from model.network import CBAM,ChannelAttention
from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models.utils import load_state_dict_from_url
# from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
import torch
from baseresnet import resnet50, resnet101
from encoder_sar import SLnet
from decoder_rgb import decoder_rgb
from collections import OrderedDict
from einops import rearrange
from torch.nn import TransformerEncoder
from model.network import context_aggregator
from typing import Optional, Dict

from torch import nn, Tensor
from torch.nn import functional as F
from data_utils import calMetric_iou, LoadDatasetFromFolder

class SmallDeepLab3(nn.Module):
    def __init__(
        self,
        branch1: nn.Module,
        branch2: nn.Module,
        classifier: nn.Module
    ) -> None:
        super().__init__()
        self.branch1 = branch1
        self.branch2 = branch2
        self.classifier = classifier
        self.cam1 = ChannelAttention(in_planes=512+160)
        self.cam2 = ChannelAttention(in_planes=3328)
        self.cbam = CBAM(in_planes=3328+512+160)
        self.conv_out = nn.Conv2d(4000,2,3)

    def forward(self, xrgb1,xrgb2,xsar) -> Dict[str, Tensor]:
        resultrgb = self.branch1(xrgb1,xrgb2)
        sardif,sarfuse = self.branch2(xsar)
        result = OrderedDict()
        xdif1 = resultrgb["xres2"].cuda()
        xconcat1 = torch.cat([xdif1, sardif], 1)
        xconcat1 = xconcat1*(self.cam1(xconcat1))
        xdif = resultrgb["xres4"].cuda()
        xconcat2 = torch.cat([xdif,sarfuse],1)
        xconcat2 = xconcat2*(self.cam2(xconcat2).cuda())
        xconcat = torch.cat([xconcat1,xconcat2],1)
        xconcat = self.cbam(xconcat).cuda()

        yfuse = self.classifier(xconcat)
        outrgb = resultrgb["out"].cuda()
        yfuse = F.interpolate(yfuse, size=[256,256], mode='bilinear', align_corners=False)
        result.update({"outrgb":outrgb,"outfuse":yfuse})
        return result

def HOLSCDnet(pretrained=True, resnet="res50", head_in_ch_rgb=2560,head_in_ch_sar=1280,head_in_ch_fuse=3328+512+160, num_classes=2):
    net = SmallDeepLab3(
        branch1 = decoder_rgb(pretrained=pretrained, resnet=resnet, head_in_ch=head_in_ch_rgb, num_classes=num_classes),
        branch2 = SLnet(in_channels=48,out_channels=head_in_ch_sar,h=64,w = 64),
        classifier=DeepLabHead(head_in_ch_fuse, num_classes)
    )
    return net


if __name__ == "__main__":
    pass