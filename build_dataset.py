import shutil
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def clean_directory(path: Path):
    """Remove all files in directory"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def process_annotations(src_dir: Path, dst_dir: Path):
    """Build annotation files for one directory"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file in tqdm(list(src_dir.glob('*.png')), desc=f'Building {src_dir.name} annotations', leave=False, mininterval=0.5):
        basename = img_file.stem
        components = basename.split('_')
        class_name = components[-1]
        
        ann_file = dst_dir / f'{basename}.txt'
        with open(ann_file, 'w') as f:
            f.write(class_name)

def main():
    parser = argparse.ArgumentParser(description='Process dataset: split and build annotations')
    parser.add_argument('--ratio', type=float, default=0.9,
                        help='Train set ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducible splitting')
    args = parser.parse_args()

    if not 0 < args.ratio < 1:
        raise ValueError(f"Ratio must be between 0 and 1, got {args.ratio}")

    # Setup paths
    src_dir = Path('dataset/rgbs')
    train_img_dir = Path('dataset/imgs/train')
    test_img_dir = Path('dataset/imgs/test')
    train_ann_dir = Path('dataset/anns/train')
    test_ann_dir = Path('dataset/anns/test')

    # Clean all output directories
    clean_directory(train_img_dir)
    clean_directory(test_img_dir)
    clean_directory(train_ann_dir)
    clean_directory(test_ann_dir)

    # Split and copy images
    rng = np.random.default_rng(args.seed)
    image_files = np.array(list(src_dir.glob('*.*')))
    rng.shuffle(image_files)

    n_train = int(len(image_files) * args.ratio)
    train_files = image_files[:n_train]
    test_files = image_files[n_train:]

    for f in tqdm(train_files, leave=False, mininterval=0.5, desc='Copying train set'):
        shutil.copy2(f, train_img_dir / f.name)
    
    for f in tqdm(test_files, leave=False, mininterval=0.5, desc='Copying test set'):
        shutil.copy2(f, test_img_dir / f.name)

    # Build annotations
    process_annotations(train_img_dir, train_ann_dir)
    process_annotations(test_img_dir, test_ann_dir)

    print(f'Processed {len(image_files)} images with seed {args.seed}:')
    print(f'Train: {len(train_files)}')
    print(f'Test: {len(test_files)}')

if __name__ == '__main__':
    main()