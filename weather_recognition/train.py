# train.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataset import WeatherDataset
from transforms import train_transform, val_test_transform
from model_baseline import SimpleCNN
from model_resnet import ResNet18Classifier
from litmodule import LitWeather

from torch.utils.data import random_split, DataLoader

data_path = "/content/drive/MyDrive/weather-recognition-dataset"

dataset = WeatherDataset(data_path, transform=train_transform)
num_classes = len(dataset.classes)

# split
train_val_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_val_size
train_val, test = random_split(dataset, [train_val_size, test_size])

val_size = int(0.2 * len(train_val))
train_size = len(train_val) - val_size
train_set, val_set = random_split(train_val, [train_size, val_size])

# apply transforms
train_set.dataset.transform = train_transform
val_set.dataset.transform = val_test_transform
test.dataset.transform = val_test_transform

train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=4)
test_loader = DataLoader(test, batch_size=512, shuffle=False, num_workers=4)

# Baseline model
baseline = SimpleCNN(num_classes)
lit_baseline = LitWeather(baseline)

trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5),
        ModelCheckpoint(save_top_k=1, monitor="val_loss", filename="best_baseline")
    ]
)

trainer.fit(lit_baseline, train_loader, val_loader)
trainer.test(lit_baseline, test_loader)