import torch
from typing import Optional, Callable
from .logger import Logger
from pprint import pprint
import datetime

__all__ = [
    'train_one_epoch',
    'val',
    'test',
]

def train_one_epoch(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_clip: Optional[float] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[Logger] = None,
        device: torch.device = torch.device("cpu"),
) -> None:
    model.train()
    if lr_scheduler is not None:
        lr_scheduler.step()
    for imgs, anns in loader:
        optimizer.zero_grad()
        imgs, anns = imgs.to(device), anns.to(device)
        preds = model(imgs)
        loss = criterion(preds, anns)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']
        if logger is not None:
            logger.update_train_log(len(loader), loss=loss.item(), lr=lr)
        else:
            pprint(f"Loss: {loss.item()} / Learning rate: {lr}")

@torch.no_grad()
def val(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        evaluator: Optional[Callable] = None,
        logger: Optional[Logger] = None,
        save_dir: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
) -> None:
    if evaluator is not None:
        model.eval()
        metrics = evaluator(model, loader, device)
        if logger is not None:
            logger.update_val_log(**metrics)
        else:
            print("Val results:")
            pprint(metrics)
        if save_dir is not None:
            reprs = [f"{k}@{round(v, 4)}" for k, v in metrics.items()]
            save_path = datetime.datetime.now().strftime("%m.%d-%H:%M:%S-") + "-".join(reprs) + '.pt'
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / save_path)


@torch.no_grad()
def test(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        evaluator: Callable,
        logger: Optional[Logger] = None,
        device: torch.device = torch.device("cpu"),
) -> None:
    model.eval()
    metrics = evaluator(model, loader, device)
    if logger is not None:
        logger.update_info(**metrics)
    else:
        print("Test results:")
        pprint(metrics)
