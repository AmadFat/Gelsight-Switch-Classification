from .tools import *
from torchvision.models import (
    resnet18 as tv_resnet18,
    ResNet18_Weights,
)
from types import MappingProxyType
from typing import Tuple


__all__ = ['get_resnet18']

Weights_ResNet18 = MappingProxyType({
    'default': ResNet18_Weights.DEFAULT,
    'imagenet1k_v1': ResNet18_Weights.IMAGENET1K_V1,
    'none': None,
})
Normlayer_ResNet18 = NormLayerType

@MODELREG.reg(name='resnet18',
              params={'num_classes': inspect_empty, 'weights': inspect_empty, 'norm_layer': "batchnorm",
                      "weight_init": "const", "bias_init": "const"},
              param_desc={"weights": ['default', 'imagenet1k_v1', 'none'], "norm_layer": ['frozenbatchnorm', 'batchnorm'],
                          "weight_init": ['const', 'xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform'],
                          "bias_init": ['const', 'xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform'],})
def get_resnet18(
        num_classes: int, weights: str, norm_layer: str,
        weight_init: str, bias_init: str,
):
    weights = Weights_ResNet18[weights.lower()]
    norm_layer = Normlayer_ResNet18[norm_layer.lower()]
    model = tv_resnet18(weights=weights, norm_layer=norm_layer)
    model.fc = MODELREG.get('replace_last_fc',
                            in_channels=model.fc.in_features, out_channels=num_classes,
                            weight_init=weight_init, bias_init=bias_init)
    return model
