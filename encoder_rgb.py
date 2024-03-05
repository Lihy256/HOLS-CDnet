from collections import OrderedDict
from model.network import context_aggregator
from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models.utils import load_state_dict_from_url
# from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
# from resnet import resnet103, resnet53
# from resnets12 import resnet103s12
from baseresnet import resnet50, resnet101
from model.network import CBAM
from collections import OrderedDict
from typing import Optional, Dict

from torch import nn, Tensor
from torch.nn import functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class backboneres(nn.Module):
    def __init__(
        self,
        backbone: nn.Module
    ) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x1: Tensor, x2: Tensor) -> Dict[str, Tensor]:
        # contract: features is a dict of tensors
        features1 = self.backbone(x1)
        features2 = self.backbone(x2)
        result = OrderedDict()
        result.update({"x1_res2":features1["res2"],"x2_res2":features2["res2"],"x1_res4":features1["res4"],"x2_res4":features2["res4"]})
        return result


def encoderres(pretrained=False, resnet="res50"):
    resnet = {
        "res50":  resnet50,
        "res101": resnet101,
    }[resnet]

    net = backboneres(
        backbone=IntermediateLayerGetter(resnet(pretrained=pretrained, replace_stride_with_dilation=[False, True, True]),
            return_layers={'layer2': 'res2', 'layer4': 'res4'}))
    #net.backbone.conv1 = nn.Conv2d(27, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return net

if __name__ == "__main__":
    import torch
    x1 = torch.randn(4,3,256,256).cuda()
    x2 = torch.randn(4,3,256,256).cuda()
    net = encoderres(True).cuda()
    print(net)
    # result = net(x1,x2)
    # for k, v in result.items():
    #     print(k, v.shape)
