from .regs import Registry, inspect_empty
from torchvision.ops import sigmoid_focal_loss
import torch


__all__ = ['CRITEREG', 'parse_criterion']


CRITEREG = Registry()

def parse_criterion(cdict: dict, *args, **kwargs):
    criterion_name = cdict.get('criterion_name') or cdict.get('criterion')
    kwargs.update({p: v for p, v in cdict.items() if CRITEREG.has_param(criterion_name, p)})
    return CRITEREG.get(criterion_name, *args, **kwargs)

@CRITEREG.reg(name='celoss',
              params={'label_smoothing': 0.0},
              param_desc={'label_smoothing': 'Float[0,1])'})
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean', label_smoothing=0.0):
        assert reduction in ['mean', 'sum', 'none'], f"reduction must be one of ['mean', 'sum', 'none']"
        assert 0 <= label_smoothing <= 1, "label_smoothing must be in [0, 1]"
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        return torch.nn.functional.cross_entropy(logits, targets,
                                                 reduction=self.reduction,
                                                 label_smoothing=self.label_smoothing)

@CRITEREG.reg(params={'gamma': inspect_empty, 'alpha': inspect_empty},
              param_desc={"alpha": "Float[0,1] or List: class weights",
                         "gamma": "Float(>=0): focus on hard samples"})
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma, alpha, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        return sigmoid_focal_loss(
            logits.log_softmax(-1), targets,
            alpha=self.alpha, gamma=self.gamma, reduction=self.reduction
        )
