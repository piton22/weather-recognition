# train.py

from pathlib import Path
import subprocess
import requests

import hydra
import pytorch_lightning as pl
import torch
from data.datamodule import WeatherDataModule
from data.dvc_utils import pull_data
from hydra.utils import instantiate
from litmodule import LitWeather
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger


def get_git_commit_id() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"
    

def mlflow_available(uri: str) -> bool:
    try:
        requests.get(uri, timeout=1)
        return True
    except Exception:
        return False


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    pull_data()
    print(OmegaConf.to_yaml(cfg))

    loggers = []

    # -------------------------
    # TensorBoard logger
    # -------------------------
    tb_logger = TensorBoardLogger(
        save_dir=cfg.logging.save_dir,
        name=cfg.logging.name,
    )
    loggers.append(tb_logger)

    # -------------------------
    # MLflow logger (только если доступен)
    # -------------------------
    if mlflow_available(cfg.mlflow.tracking_uri):
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.tracking_uri,
        )
        # логируем commit id и гиперпараметры
        mlflow_logger.experiment.log_param(
            mlflow_logger.run_id,
            "git_commit_id",
            get_git_commit_id(),
        )
        mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        loggers.append(mlflow_logger)
        print("[INFO] MLflow logger enabled")
    else:
        print("[WARN] MLflow server not available. Skipping MLflow logging.")
        mlflow_logger = None


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

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        callbacks=callbacks,
        logger=loggers,
    )

    # --- Train & Test ---
    trainer.fit(lit_model, dm)
    trainer.test(lit_model, dm)

    # --- Save pure torch model ---
    save_path = cfg.train.save_path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"\nМодель сохранена в: {save_path}\n")


if __name__ == "__main__":
    main()
