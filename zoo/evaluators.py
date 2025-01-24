from .regs import Registry, inspect_empty
from typing import Callable, Sequence
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

    def __call__(self, model, loader, device):
        results = {}
        for e in self.evaluators:
            results.update(e(model, loader, device))
        return results


@EVALREG.reg()
class Accuracy:
    def __call__(self, model, loader, device):
        correct, total = 0, 0
        for imgs, anns in tqdm(loader, leave=False, desc="Accuracy", mininterval=0.5):
            imgs, anns = imgs.to(device), anns.to(device)
            preds = model(imgs).argmax(1)
            anns = anns.argmax(1)
            correct += (preds == anns).sum().item()
            total += anns.shape[0]
        return {'Accuracy': 100 * correct / total}


@EVALREG.reg(params={'criterion': inspect_empty},
             param_desc={'criterion': 'Callable',})
class Loss:
    def __init__(self, criterion: Callable):
        assert callable(criterion)
        self.criterion = criterion

    def __call__(self, model, loader, device):
        loss = 0
        for imgs, anns in tqdm(loader, leave=False, desc="Loss", mininterval=0.5):
            imgs, anns = imgs.to(device), anns.to(device)
            preds = model(imgs)
            loss += self.criterion(preds, anns).item()
        return {'Loss': loss / len(loader)}
