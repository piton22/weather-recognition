# train.py

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from data.datamodule import WeatherDataModule
from hydra.utils import instantiate
from litmodule import LitWeather
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # --- DataModule ---
    dm = WeatherDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_val_split=cfg.data.train_val_split,
        val_split_within=cfg.data.val_split_within,
    )
    dm.setup()

    # --- Set number of classes ---
    cfg.model.num_classes = len(dm.classes)

    # --- Instantiate model ---
    model = instantiate(cfg.model)
    lit_model = LitWeather(model)

    # --- Callbacks ---
    callbacks = [
        EarlyStopping(
            monitor=cfg.train.early_stopping.monitor,
            patience=cfg.train.early_stopping.patience,
        ),
        ModelCheckpoint(
            save_top_k=cfg.train.checkpoint.save_top_k,
            monitor=cfg.train.checkpoint.monitor,
            filename=cfg.train.checkpoint.filename,
        ),
    ]

    trainer = pl.Trainer(max_epochs=cfg.train.max_epochs, callbacks=callbacks)

    # --- Train & Test ---
    trainer.fit(lit_model, dm)
    trainer.test(lit_model, dm)

    # ----------------------------------------------------
    # Сохранение модели вручную в путь из config
    # ----------------------------------------------------
    save_path = cfg.train.save_path

    # создаём директорию, если её нет
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # сохраняем state_dict чистой модели, без Lightning-обёртки
    torch.save(model.state_dict(), save_path)

    print(f"\n Модель сохранена в: {save_path}\n")


if __name__ == "__main__":
    main()
