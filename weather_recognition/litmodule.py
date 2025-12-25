from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn.functional as fnc
from torchmetrics.classification import MulticlassF1Score


@dataclass
class ModelConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    num_classes: int = 11


class LitWeather(pl.LightningModule):
    def __init__(self, model, cfg: ModelConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg

        self.train_f1 = MulticlassF1Score(num_classes=self.cfg.num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=self.cfg.num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=self.cfg.num_classes, average="macro")

        self.save_hyperparameters(ignore=["model", "cfg"])

    def forward(self, images):
        return self.model(images)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.cfg.scheduler_patience,
            factor=self.cfg.scheduler_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def step(self, batch):
        images, labels = batch
        logits = self(images)
        loss = fnc.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        return loss, acc, preds

    def training_step(self, batch, batch_idx):
        loss, acc, preds = self.step(batch)
        labels = batch[1]

        self.train_f1.update(preds, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        f1 = self.train_f1.compute()
        self.log("train_f1", f1, on_epoch=True, prog_bar=True)
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        loss, acc, preds = self.step(batch)
        labels = batch[1]

        self.val_f1.update(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        loss, acc, preds = self.step(batch)
        labels = batch[1]

        self.test_f1.update(preds, labels)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    def on_test_epoch_end(self):
        f1 = self.test_f1.compute()
        self.log("test_f1", f1, on_epoch=True)
        self.test_f1.reset()
