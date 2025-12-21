from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

valid_extensions = {".jpg", ".jpeg", ".png"}


class WeatherDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(entry.name for entry in Path(root_dir).iterdir() if entry.is_dir())

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        for cls in self.classes:
            cls_dir = Path(root_dir) / cls
            for path in cls_dir.iterdir():
                if path.suffix.lower() in valid_extensions:
                    self.samples.append((cls_dir / path.name, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            return torch.zeros(3, 224, 224), label
