import pytorch_lightning as pl
import torch
import torch.nn.functional as fnc


class LitWeather(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["model"])

    def forward(self, images):
        return self.model(images)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
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
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)

        # --- логирование для MLflow ---
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
