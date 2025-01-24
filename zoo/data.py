from torch.utils.data import (
    Dataset as tDataset, DataLoader, random_split,
    RandomSampler, SequentialSampler, BatchSampler
)
from typing import Callable, Optional, Tuple
from torch.nn.functional import one_hot
from .transforms import TRANSREG
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch


__all__ = [
    'get_train_val_loaders',
    'get_test_loader',
    'set_seed',
]

def set_seed(seed: int, deterministic: bool = False):
    if seed is not None:
        import random
        import numpy
        import torch
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = deterministic
                torch.backends.cudnn.benchmark = not deterministic
        elif deterministic:
            torch.set_num_threads(1)


class Dataset(tDataset):
    def __init__(
            self,
            root: str,
            istrain: bool,
            transform: Optional[Callable] = None):
        img_dir = Path(root) / 'imgs' / ('train' if istrain else 'test')
        ann_dir = Path(root) / 'anns' / ('train' if istrain else 'test')
        self.img_paths = sorted(img_dir.glob("*"))
        self.ann_paths = sorted(ann_dir.glob("*.txt"))
        self.set_transform(transform)

        if len(self.img_paths) != len(self.ann_paths):
            print("Number of images and annotations do not match")

        print("Checking pair match...")
        for img_path, ann_path in zip(self.img_paths, self.ann_paths):
            assert img_path.stem == ann_path.stem, f"Wrong match: {img_path.stem} and {ann_path.stem}"
        print(f"All pairs match: {len(self.img_paths)} pairs")

        print("Prepare vocabulary...")
        vocabulary, ann_dir = set(), Path(root) / 'anns'
        for ann_path in tqdm((ann_dir / 'train').glob("*.txt"), leave=False, mininterval=0.5):
            vocabulary.add(ann_path.read_text().splitlines()[0].lower())
        for ann_path in tqdm((ann_dir / 'test').glob("*.txt"), leave=False, mininterval=0.5):
            vocabulary.add(ann_path.read_text().splitlines()[0].lower())
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocabulary))}
        self.dictionary = {idx: word for word, idx in self.vocabulary.items()}
        print(f"Vocabulary prepared - {len(self.vocabulary)} words: {self.vocabulary}")

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.img_paths[idx]).convert("RGB"))
        ann = self.vocabulary[self.ann_paths[idx].read_text().splitlines()[0].lower()]
        return img, ann
    
    def __len__(self):
        return len(self.img_paths)

    def set_transform(self, transform: Optional[Callable] = None):
        self.transform = transform if transform is not None else TRANSREG.get('totensor')
    
    def split(
            self,
            ratio_train: float,
            ratio_val: float,
            transform_train: Optional[Callable] = None,
            transform_val: Optional[Callable] = None,
    ):
        assert ratio_train + ratio_val == 1.0
        data_train, data_val = random_split(self, (ratio_train, ratio_val))
        data_train.dataset.set_transform(transform_train)
        data_val.dataset.set_transform(transform_val)
        return data_train, data_val


def get_collate_fn(num_classes: int):
    def collate_fn(batch):
        imgs, anns = [], []
        for x in batch:
            imgs.append(x[0])
            anns.append(x[1])
        imgs = torch.stack(imgs)
        anns = torch.tensor(anns)
        anns = one_hot(anns, num_classes)
        return imgs, anns.float()
    return collate_fn


def get_train_val_loaders(
        root: str,
        transform_train: Optional[Callable],
        transform_val: Optional[Callable],
        split_ratio: Tuple[float, float],
        batch_size: int,
        num_workers: int = 0,
        seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, int, dict]:
    data = Dataset(root=root, istrain=True)
    num_classes = len(data.vocabulary)
    data_train, data_val = data.split(*split_ratio,
                                      transform_train=transform_train,
                                      transform_val=transform_val,)
    collate_fn = get_collate_fn(num_classes)
    generator = torch.Generator().manual_seed(seed) if seed else None
    sampler_train = RandomSampler(data_train, generator=generator)
    sampler_train = BatchSampler(sampler_train, batch_size, drop_last=True)
    loader_train = DataLoader(data_train,
                              batch_sampler=sampler_train,
                              collate_fn=collate_fn,
                              num_workers=num_workers)
    sampler_val = SequentialSampler(data_val)
    sampler_val = BatchSampler(sampler_val, 1, drop_last=False)
    loader_val = DataLoader(data_val,
                            batch_sampler=sampler_val,
                            collate_fn=collate_fn,
                            num_workers=num_workers)
    return loader_train, loader_val, num_classes, data.dictionary


def get_test_loader(
        root: str,
        transform: Optional[Callable],
        batch_size: int = 1,
        num_workers: int = 0,
) -> Tuple[DataLoader, int, dict]:
    data = Dataset(root=root, istrain=False, transform=transform)
    num_classes = len(data.vocabulary)
    collate_fn = get_collate_fn(num_classes)
    sampler = SequentialSampler(data)
    sampler = BatchSampler(sampler, batch_size, drop_last=False)
    loader = DataLoader(data,
                        batch_sampler=sampler,
                        collate_fn=collate_fn,
                        num_workers=num_workers)
    return loader, num_classes, data.dictionary
