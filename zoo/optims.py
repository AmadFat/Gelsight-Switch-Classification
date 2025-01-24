from .regs import Registry, inspect_empty
from torch.optim import (
    SGD as tSGD,
    AdamW as tAdamW,
)
from torch.optim.lr_scheduler import (
    ConstantLR as tConstLR,
    MultiStepLR as tStepLR,
    CosineAnnealingWarmRestarts as tCosineLR,
)


__all__ = ['OPTIMREG', 'SCHEDREG', 'parse_optimizer', 'parse_scheduler']


OPTIMREG = Registry()
SCHEDREG = Registry()

def parse_optimizer(odict: dict, *args, **kwargs):
    optimizer_name = odict.get('optimizer_name') or odict.get('optimizer')
    kwargs.update({p: v for p, v in odict.items() if OPTIMREG.has_param(optimizer_name, p)})
    return OPTIMREG.get(optimizer_name, *args, **kwargs)

def parse_scheduler(sdict: dict, *args, **kwargs):
    scheduler_name = sdict.get('scheduler_name') or sdict.get('scheduler')
    kwargs.update({p: v for p, v in sdict.items() if SCHEDREG.has_param(scheduler_name, p)})
    return SCHEDREG.get(scheduler_name, *args, **kwargs)

SGD = OPTIMREG.reg(name='sgd',
                   params={'lr': inspect_empty, 'momentum': 0.95, 'weight_decay': 1e-5},
                   param_desc={'lr': 'Learning rate', 'momentum': '[0, 1]', 'weight_decay': 'Weight decay'})(tSGD)

AdamW = OPTIMREG.reg(name='adamw',
                     params={'lr': inspect_empty, 'betas': (0.9, 0.999), 'weight_decay': 1e-5},
                     param_desc={'lr': 'Learning rate', 'betas': 'Tuple[Float[0,1], Float[0,1]]', 'weight_decay': 'Weight decay'})(tAdamW)

ConstLR = SCHEDREG.reg(name='constlr',
                       params={"factor": 0.1},
                       param_desc={"factor": "Float[0,1]: initial decay factor"})(tConstLR)

StepLR = SCHEDREG.reg(name='steplr',
                      params={"milestones": inspect_empty, "gamma": 0.3162},
                      param_desc={"milestones": "Sequence[Int]: decay epoch milestones",
                                  "gamma": "Float[0,1]: decay factor 0 ~ 1"})(tStepLR)

CosineLR = SCHEDREG.reg(name='cosinelr',
                        params={"T_0": inspect_empty, "T_mult": 1, "eta_min": 0},
                        param_desc={"T_0": "Int: basic decay period",
                                    "T_mult": "Int: period multiplier",
                                    "eta_min": "Float: minimum learning rate"})(tCosineLR)
