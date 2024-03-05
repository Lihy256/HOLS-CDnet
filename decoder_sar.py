from collections import OrderedDict

from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models.utils import load_state_dict_from_url
# from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

from baseresnet import resnet50, resnet101
from encoder_sar import FLBCDnet
from collections import OrderedDict
from typing import Optional, Dict

from torch import nn, Tensor
from torch.nn import functional as F


class SmallDeepLab(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        x = self.backbone(x)
        resultsar = OrderedDict()
        x = x[1]
        y = self.classifier(x)
        y = F.interpolate(y, size=[256,256], mode='bilinear', align_corners=False)
        resultsar.update({"outsar":y,"midsar":x})
        return resultsar


def decoder_sar(head_in_ch=1280, num_classes=2):
    net = SmallDeepLab(
        backbone=FLBCDnet(in_channels=48,out_channels=1280,h=64,w = 64),
        classifier=DeepLabHead(head_in_ch, num_classes)
    )
    return net

if __name__ == "__main__":
    import torch
    x = torch.randn(2,48,64,64).cuda()
    net = decoder_sar().cuda()
    result = net(x)
    y = result['outsar']
    print(y.shape)
