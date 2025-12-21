import pytorch_lightning as pl
import torch
from data.dataset import WeatherDataset
from data.transforms import train_transform, val_test_transform
from torch.utils.data import DataLoader, random_split


class WeatherDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=512,
        num_workers=4,
        train_val_split=0.8,
        val_split_within=0.2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_val_split = train_val_split
        self.val_split_within = val_split_within

        self.dataset = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.classes = None

    def setup(self, stage=None):
        # Загружаем датасет
        self.dataset = WeatherDataset(self.data_dir, transform=train_transform)
        self.classes = self.dataset.classes

        # Разделяем train+val и test
        train_val_size = int(self.train_val_split * len(self.dataset))
        test_size = len(self.dataset) - train_val_size
        train_val, test = random_split(
            self.dataset, [train_val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

        # Разделяем train и val
        val_size = int(self.val_split_within * len(train_val))
        train_size = len(train_val) - val_size

        self.train_set, self.val_set = random_split(
            train_val, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # Назначаем трансформации
        self.train_set.dataset.transform = train_transform
        self.val_set.dataset.transform = val_test_transform
        test.dataset.transform = val_test_transform

        self.test_set = test

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
