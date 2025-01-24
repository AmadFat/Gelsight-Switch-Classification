from ..regs import *
from torchvision.ops import FrozenBatchNorm2d

from types import MappingProxyType
from functools import partial
from typing import Optional
import torch

__all__ = ['replace_last_fc', 'MODELREG', 'inspect_empty', 'NormLayerType', 'parse_model']

MODELREG = Registry()

def parse_model(mdict: dict, *args, **kwargs):
    model_name = mdict.get('model_name') or mdict.get('model')
    kwargs.update({p: v for p, v in mdict.items() if MODELREG.has_param(model_name, p)})
    return MODELREG.get(model_name, *args, **kwargs)

# Supported initialization methods
InitType = MappingProxyType({
    "notevenconst": partial(torch.nn.init.normal_, mean=0, std=1e-2),
    "const_weight": partial(torch.nn.init.constant_, val=1),
    "const_bias": partial(torch.nn.init.constant_, val=0),
    "xavier_normal": torch.nn.init.xavier_normal_,
    "xavier_uniform": torch.nn.init.xavier_uniform_,
    "kaiming_normal": torch.nn.init.kaiming_normal_,
    "kaiming_uniform": torch.nn.init.kaiming_uniform_,
})

NormLayerType = MappingProxyType({
    "frozenbatchnorm": FrozenBatchNorm2d,
    "batchnorm": torch.nn.BatchNorm2d,
})

def _init_pt_(
        pt: torch.Tensor,
        init_method: str,
):
    if init_method.lower().startswith(("xavier", "kaiming")) and pt.ndim < 2:
        InitType['notevenconst'](pt)
    else:
        InitType[init_method.lower()](pt)

@MODELREG.reg(params={"in_channels": inspect_empty, "out_channels": inspect_empty, "use_bias": True, "weight_init": "const", "bias_init": "const"},
              param_desc={"weight_init": ["const", "xavier_normal", "xavier_uniform", "kaiming_normal", "kaiming_uniform"],
                          "bias_init": ["const", "xavier_normal", "xavier_uniform", "kaiming_normal", "kaiming_uniform"]})
def replace_last_fc(
        in_channels: int,
        out_channels: int,
        use_bias: bool,
        weight_init: str,
        bias_init: Optional[str] = None,
) -> torch.nn.Linear:
    fc = torch.nn.Linear(in_channels, out_channels, bias=use_bias)
    _init_pt_(fc.weight, 'const_weight' if weight_init.lower() == "const" else weight_init)
    if use_bias:
        _init_pt_(fc.bias, 'const_bias' if use_bias and bias_init.lower() == "const" else bias_init)
    return fc