import os
from pathlib import Path

print("Working directory: ", Path(__file__).parent)
data_dir = Path(__file__).parent / 'dataset'
data_dir.mkdir(parents=True, exist_ok=True)
img_dir = data_dir / 'imgs'
img_dir.mkdir(parents=True, exist_ok=True)
ann_dir = data_dir / 'anns'
ann_dir.mkdir(parents=True, exist_ok=True)
img_train_dir = img_dir / 'train'
img_test_dir = img_dir / 'test'
img_train_dir.mkdir(parents=True, exist_ok=True)
img_test_dir.mkdir(parents=True, exist_ok=True)
ann_train_dir = ann_dir / 'train'
ann_test_dir = ann_dir / 'test'
ann_train_dir.mkdir(parents=True, exist_ok=True)
ann_test_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir = Path(__file__).parent / 'ckpts'
ckpt_dir.mkdir(parents=True, exist_ok=True)
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
tensorboard_dir = data_dir.parent / 'tbevents'
tensorboard_dir.mkdir(parents=True, exist_ok=True)

try: 
    import numpy
except:
    print("Numpy is not installed.")
    print("Installing Numpy...")
    os.system("pip install 'numpy<2' -q")

import numpy
# check numpy version < 2
if int(numpy.__version__.split('.')[0]) > 1:
    print("Numpy version incompatibility detected.")
    os.system("pip install 'numpy<2' -q")
    print("Numpy version is updated to 1.x")
print("Numpy is correctly installed.")

# check pytorch is correctly installed
try:
    import torch, torchvision, torchaudio
except:
    "PyTorch is recommended to be manually installed."
    exit()
print("PyTorch is correctly installed.")

try:
    import tabulate
except:
    print("Tabulate is not installed.")
    print("Installing Tabulate...")
    os.system("pip install tabulate -q")
print("Tabulate is correctly installed.")

try:
    import tqdm
except:
    print("Tqdm is not installed.")
    print("Installing Tqdm...")
    os.system("pip install tqdm -q")
print("Tqdm is correctly installed.")


try:
    import torchsummary
except:
    print("Torchsummary is not installed.")
    print("Installing Torchsummary...")
    os.system("pip install torchsummary -q")
print("Torchsummary is correctly installed.")


try:
    import tensorboard
except:
    print("Tensorboard is not installed.")
    print("Installing Tensorboard...")
    os.system("pip install tensorboard -q")
print("Tensorboard is correctly installed.")


try:
    import psutil
except:
    print("Psutil is not installed.")
    print("Installing Psutil...")
    os.system("pip install psutil -q")
print("Psutil is correctly installed.")

try:
    import yaml
except:
    print("PyYaml is not installed.")
    print("Installing PyYaml...")
    os.system("pip install pyyaml -q")
print("PyYaml is correctly installed.")