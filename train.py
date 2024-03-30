import argparse
import random
from pathlib import Path

import pytorch_lightning as lightning
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers

import models
import wandb
from dataloaders import dataloader_v7

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distance Estimator Training")
    parser.add_argument(
        "--experiment_name", type=str, help="Name of the experiment", default="exp0"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path of the dataset",
        default="/mnt/carla_capture/",
    )
    parser.add_argument(
        "--continue_from", type=str, help="Path of the checkpoint", default=None
    )
    parser.add_argument(
        "--logger_path", type=str, help="Path of the logger", default=None
    )
    parser.add_argument(
        "--gpus", type=str, help="Number of GPUs to use, example: 0,3,4", default=None
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    dataset_root = Path(args.dataset_path)
    # ida rgb: model v23, dataloader v8
    # dearsan on: model v24, dataloader v7
    # ida thermal: model v25, dataloader v9
    model = models.DistNet_v33()
    datamodule = dataloader_v7.DepthDataModule(
        dataset_root / "cam_rgb",
        dataset_root / "labels",
        dataset_root / "true_distance_not_z",
        # dataset_root / "yolo_inferences" # FOR V8
        # dataset_root / "features", # old tryings
    )

    logger_path = args.logger_path or root / "logs"
    tensorboard_logger = loggers.TensorBoardLogger(
        logger_path, name=args.experiment_name
    )
    # wandb.init(project="DistanceEstimationNetwork", sync_tensorboard=True)

    trainer = lightning.Trainer(
        logger=tensorboard_logger,
        accelerator="gpu",
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=54),
            callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=4),
        ],
        max_epochs=90,
        devices=args.gpus,
        strategy="ddp" if args.gpus else "auto",
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.continue_from,
    )
