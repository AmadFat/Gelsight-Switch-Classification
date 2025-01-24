from .tools import *
from torchvision.models import (
    mobilenet_v3_small as tv_mobilenet_v3_small,
    MobileNet_V3_Small_Weights as TV_Weights_MobileNet_V3_Small,
)
from types import MappingProxyType


__all__ = ['get_mobilenet_v3_small']

Weights_MobileNet_V3_Small = MappingProxyType({
    'default': TV_Weights_MobileNet_V3_Small.DEFAULT,
    'imagenet1k_v1': TV_Weights_MobileNet_V3_Small.IMAGENET1K_V1,
    'none': None,
})
Normlayer_MobileNet_V3_Small = NormLayerType

@MODELREG.reg(name='mobilenet_v3_s',
              params={'num_classes': inspect_empty, 'weights': inspect_empty, 'norm_layer': "batchnorm",
                      "weight_init": "const", "bias_init": "const", "dropout": 0.2},
              param_desc={"weights": ['default', 'imagenet1k_v1', 'none'], "norm_layer": ['frozenbatchnorm', 'batchnorm'], "dropout": '0 ~ 1',
                          "weight_init": ['const', 'xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform'],
                          "bias_init": ['const', 'xavier_normal', 'xavier_uniform', 'kaiming_normal', 'kaiming_uniform'],
                          })
def get_mobilenet_v3_small(
        num_classes: int, weights: str, norm_layer: str,
        weight_init: str, bias_init: str, dropout: float,
):
    weights = Weights_MobileNet_V3_Small[weights.lower()]
    norm_layer = Normlayer_MobileNet_V3_Small[norm_layer.lower()]
    model = tv_mobilenet_v3_small(weights=weights, norm_layer=norm_layer, dropout=dropout)
    model.classifier[3] = MODELREG.get('replace_last_fc',
                                       in_channels=model.classifier[3].in_features, out_channels=num_classes,
                                       weight_init=weight_init, bias_init=bias_init)
    return model
