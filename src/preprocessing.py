import io
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

IMG_SIZE = 224
# alphabetical so it matches how ImageFolder assigns label indices
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet stats, since the backbone is pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloader(data_dir, batch_size=32, train=True, shuffle=None, num_workers=0):
    dataset = datasets.ImageFolder(str(data_dir), transform=get_transforms(train))
    if shuffle is None:
        shuffle = train
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


def get_combined_loader(dirs, batch_size=32, train=True, num_workers=0):
    tfm = get_transforms(train)
    parts = []
    for d in dirs:
        d = Path(d)
        if d.exists() and any(d.rglob("*.jpg")):
            # allow_empty keeps class_to_idx over all classes even when the
            # uploads folder only has images for some of them
            parts.append(datasets.ImageFolder(str(d), transform=tfm, allow_empty=True))
    if not parts:
        raise ValueError("no images to load")
    data = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    return DataLoader(data, batch_size=batch_size, shuffle=train, num_workers=num_workers)


def ensure_class_dirs(root):
    root = Path(root)
    for c in CLASS_NAMES:
        (root / c).mkdir(parents=True, exist_ok=True)
    return root


def preprocess_image(image):
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    image = image.convert("RGB")
    return get_transforms(train=False)(image).unsqueeze(0)
