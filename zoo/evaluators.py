from .regs import Registry, inspect_empty
from typing import Callable, Sequence, Optional
from collections import defaultdict
from tqdm import tqdm

__all__ = ['EVALREG', 'parse_evaluator']


EVALREG = Registry()

def parse_evaluator(edict: dict, *args, **kwargs):
    evaluators = []
    if 'acc' in edict:
        evaluators.append(EVALREG.get('accuracy'))
    if 'loss' in edict:
        evaluators.append(EVALREG.get('loss', criterion=kwargs.get('criterion')))
    return EVALREG.get('compose', evaluators=evaluators)


@EVALREG.reg(params={'evaluators': inspect_empty},
             param_desc={'evaluators': 'Sequence[Callable]',})
class Compose:
    def __init__(self, evaluators: Sequence[Callable]):
        assert all(callable(e) for e in evaluators)
        self.evaluators = evaluators

    def __call__(self, model, loader, device, dictionary: Optional[dict] = None):
        results = {}
        for e in self.evaluators:
            results.update(e(model, loader, device, dictionary=dictionary))
        return results


@EVALREG.reg()
class Accuracy:
    def __call__(self, model, loader, device, dictionary: Optional[dict] = None):
        acc_dict = defaultdict(lambda: {'correct': 0, 'total': 0})
        for imgs, anns in tqdm(loader, leave=False, desc="Accuracy", mininterval=0.5):
            imgs, anns = imgs.to(device), anns.to(device)
            num_classes = anns.shape[1]
            preds = model(imgs).argmax(1)
            anns = anns.argmax(1)
            for idx in range(num_classes):
                acc_dict[idx]['correct'] += (preds == anns & anns == idx).sum().item()
                acc_dict[idx]['total'] += (anns == idx).sum().item()
        if dictionary is not None:
            acc_dict = {dictionary.get(k, k): v for k, v in acc_dict.items()}
            acc_dict['avg'] = {'correct': 0, 'total': 0}
        acc_dict['avg']['correct'] = sum([v['correct'] for v in acc_dict.values()])
        acc_dict['avg']['total'] = sum([v['total'] for v in acc_dict.values()])
        return {"acc_" + str(k): 100 * v['correct'] / (v['total'] + 1e-6) for k, v in acc_dict.items()}


@EVALREG.reg(params={'criterion': inspect_empty},
             param_desc={'criterion': 'Callable',})
class Loss:
    def __init__(self, criterion: Callable):
        assert callable(criterion)
        self.criterion = criterion

    def __call__(self, model, loader, device, dictionary: Optional[dict] = None):
        loss = 0
        for imgs, anns in tqdm(loader, leave=False, desc="Loss", mininterval=0.5):
            imgs, anns = imgs.to(device), anns.to(device)
            preds = model(imgs)
            loss += self.criterion(preds, anns).item()
        return {'Loss': loss / len(loader)}
