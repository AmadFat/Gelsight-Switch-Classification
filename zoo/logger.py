from torch.utils.tensorboard import SummaryWriter
from collections import deque, defaultdict
from tabulate import tabulate
from typing import Optional
from pathlib import Path
import datetime
import time


__all__ = ['Logger']

def get_sync_time():
    return datetime.datetime.now().strftime("%H:%M:%S %m-%d")

def get_tabulate(kwargs: dict):
    return tabulate(kwargs.items(),
                    tablefmt="grid",
                    colalign=["right", "right"],
                    stralign=["right", "right"],)


class Logger:
    def __init__(
            self,
            train_print_interval: int = 1,
            window_metric: int = 1,
            window_time_stamp: int = 1,
            log_save_path: Optional[str] = None,
            use_tensorboard: bool = False,
            tb_save_path: Optional[str] = None,
    ):
        self.local_tracer = defaultdict(lambda: deque(maxlen=window_metric))
        self.global_tracer = defaultdict(lambda: {"v": 0., "n": 0.})
        self.time_deltas = deque(maxlen=window_time_stamp)
        self.last_timestamp = None
        self.log_interval = train_print_interval
        self.save_path = Path(log_save_path) if log_save_path else None
        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path.touch(exist_ok=False)
        self.train_epoch_idx, self.train_step_idx, self.val_epoch_idx = 1, 0, 1
        self.update_log(f"Log file created at {self._get_sync_time()} with path: {self.save_path}")
        self.update_log("System information:")
        self.update_info(**get_system_info())

        # Tensorboard
        if use_tensorboard and tb_save_path is not None:
            # Create the tbevents folder next to the log file
            tb_events_dir = Path(tb_save_path)
            tb_events_dir.mkdir(parents=True, exist_ok=True)
            self.tb = SummaryWriter(tb_events_dir)
            self.tb_idx = 0
        else:
            self.tb = None

    def update_log(self, log: str):
        print(log)
        if self.save_path is not None and self.save_path.exists():
            with open(self.save_path, 'a') as f:
                f.write(log + "\n")

    def update_train_log(self, max_steps, **kwargs):
        self.train_step_idx += 1
        self.val_epoch_idx = self.train_epoch_idx
        
        # Update timing
        current_time = time.perf_counter()
        if self.last_timestamp is not None:
            self.time_deltas.append(current_time - self.last_timestamp)
        self.last_timestamp = current_time

        for m, v in kwargs.items():
            self.local_tracer[m].append(v)
            self.global_tracer[m]["v"] += v
            self.global_tracer[m]["n"] += 1
            
        if self.train_step_idx in [1, max_steps] or self.train_step_idx % self.log_interval == 0:
            current_time = self._get_sync_time()
            epoch_idx = self.train_epoch_idx
            step_idx = f"{self.train_step_idx}/{max_steps}"
            eta = self._calculate_eta(max_steps)
            info = f"{current_time} [epoch {epoch_idx} train] [step {step_idx}] [eta {eta}]: "
            metrics = []
            for k in set(self.global_tracer.keys()).union(self.local_tracer.keys()):
                if k in self.global_tracer and k in self.local_tracer:
                    v_global = self.global_tracer[k]["v"] / (self.global_tracer[k]["n"] + 1e-6)
                    v_local = sum(self.local_tracer[k]) / len(self.local_tracer[k])
                    # Fix the condition for lr formatting
                    if k in ['lr', 'learning_rate']:
                        m = f"{k} {v_local:.4e}"
                    else:
                        m = f"{k} {v_global:.4f}({v_local:.4f})"
                    metrics.append(m)
            self.update_log(info + " | ".join(metrics))
            
        if self.train_step_idx == max_steps:
            self.train_step_idx = 0
            self.train_epoch_idx += 1
            self.global_tracer.clear()

        if self.tb is not None:
            self.tb_idx += 1
            for k, v in kwargs.items():
                self.tb.add_scalar(f"train/{k}", v, self.tb_idx)

    def update_val_log(self, **kwargs):
        info = f"{self._get_sync_time()} [epoch {self.val_epoch_idx} val]: "
        metrics = [f"{k} {v:.4f}" for k, v in kwargs.items()]
        self.update_log(info + " | ".join(metrics))

        if self.tb is not None:
            for k, v in kwargs.items():
                self.tb.add_scalar(f"val/{k}", v, self.tb_idx)

    def update_info(self, **kwargs):
        self.update_log(self._get_tabulate(kwargs))

    def update_model_graph(self, model, input_shape=(1, 3, 224, 224)):
        if self.tb is not None:
            import torch
            input = torch.randn(input_shape).to(next(model.parameters()).device)
            self.tb.add_graph(model, input)

    def close(self):
        if self.tb is not None:
            self.tb.close()
        del self

    def _calculate_eta(self, max_steps):
        if not self.time_deltas:
            return "N/A"
        avg_step_time = sum(self.time_deltas) / len(self.time_deltas)
        remain_steps = max_steps - self.train_step_idx
        eta_seconds = avg_step_time * remain_steps
        return str(datetime.timedelta(seconds=int(eta_seconds)))

    @staticmethod
    def _get_sync_time():
        return get_sync_time()
    
    @staticmethod
    def _get_tabulate(kwargs: dict):
        return get_tabulate(kwargs)

def get_system_info():
    """Get detailed system information grouped by category"""
    # Hardware info
    import psutil
    import platform
    import torch, torchvision, torchaudio
    hardware_info = {
        'CPU Physical Cores': psutil.cpu_count(logical=False),
        'CPU Logical Cores': psutil.cpu_count(logical=True),
        'CPU Max Frequency': f"{psutil.cpu_freq().max:.2f} MHz",
        'CPU Current Frequency': f"{psutil.cpu_freq().current:.2f} MHz",
        'RAM Total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        'RAM Available': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        'RAM Used': f"{psutil.virtual_memory().used / (1024**3):.2f} GB",
    }

    # Software info
    software_info = {
        'Hostname': platform.node(),
        'OS': platform.system(),
        'OS Release': platform.release(),
        'OS Version': platform.version(),
        'Python Version': platform.python_version(),
        'PyTorch Version': torch.__version__,
        'TorchVision Version': torchvision.__version__,
        'TorchAudio Version': torchaudio.__version__,
    }

    # GPU info if available
    if torch.cuda.is_available():
        hardware_info.update({
            'GPU Device': torch.cuda.get_device_name(),
            'GPU Count': torch.cuda.device_count(),
            'GPU Memory Total': f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB",
            'GPU Memory Reserved': f"{torch.cuda.memory_reserved(0) / (1024**3):.2f} GB",
            'CUDA Version': torch.version.cuda,
        })
    return {**hardware_info, **software_info}


if __name__ == '__main__':
    import time
    train_print_interval = 3
    window_metric, window_time_stamp = 3, 3
    max_epochs, max_steps = 10, 10
    logger = Logger(train_print_interval=train_print_interval,
                         save_path='test.log',
                         window_metric=window_metric,
                         window_time_stamp=window_time_stamp,
                         use_tensorboard=False)
    for _ in range(max_epochs):
        for _ in range(1, max_steps + 1):
            time.sleep(1)
            logger.update_train_log(int(max_steps), loss=_, acc=0.9)
        logger.update_val_log(loss=0.2, acc=0.8)
    logger.close()