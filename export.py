import argparse
import random
from pathlib import Path

import pytorch_lightning as lightning
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
import torch

import models
import wandb
from dataloaders import dataloader_v7

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distance Estimator ONNX Export")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path of the dataset",
        default="/mnt/carla_capture/",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path of the checkpoint", default=None
    )
    parser.add_argument("--onnx", type=str, help="Save path of the onnx", default=None)
    args = parser.parse_args()

    root = Path(__file__).parent
    dataset_root = Path(args.dataset_path)
    model = models.DistNet_v30.load_from_checkpoint(args.checkpoint)
    datamodule = dataloader_v7.DepthDataModule(
        dataset_root / "cam_rgb",
        dataset_root / "labels",
        dataset_root / "true_distance_not_z",
    )
    # datamodule = dataloader_v4.DepthDataModule(
    #     dataset_root / "cam_rgb",
    #     dataset_root / "labels",
    #     dataset_root / "true_distance_not_z",
    #     dataset_root / "features",
    # )
    # tensorboard_logger = loggers.TensorBoardLogger(
    #     root / "logs", name=args.experiment_name
    # )
    # wandb.init(project="DistanceEstimationNetwork", sync_tensorboard=True)

    # trainer = lightning.Trainer(
    #     logger=tensorboard_logger,
    #     accelerator="gpu",
    #     devices=1,
    #     callbacks=[
    #         callbacks.EarlyStopping(monitor="val_loss", patience=54),
    #         callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=4),
    #     ],
    #     max_epochs=200,
    # )

    # trainer.fit(
    #     model,
    #     datamodule=datamodule,
    #     ckpt_path=args.continue_from,
    # )

    datamodule.setup("val")
    roi, bb_with_class, target_distance = next(iter(datamodule.train_dataloader()))
    roi, bb_with_class, target_distance = (
        roi.cuda(),
        bb_with_class.cuda(),
        target_distance.cuda(),
    )
    model.eval()
    print(roi.shape, bb_with_class.shape, target_distance.shape)

    # import time
    for _ in range(1):
        # start = time.perf_counter()

        (
            predicted_distance,
            # predicted_distance_blackbox,
            # predicted_distance_height_priors,
            # predicted_distance_raytracing,
            # combined_coefficients,
            # predicted_distance_raytracing_classical,
        ) = model(roi, bb_with_class)

    # end = time.perf_counter()
    # print(f"%{(end-start)*1000} ms")

    print(predicted_distance.shape)

    # model.to_onnx(args.onnx, (roi, bb_with_class), export_params=True)
    torch.onnx.export(
        model,  # model being run
        ##since model is in the cuda mode, input also need to be
        (roi, bb_with_class),  # model input (or a tuple for multiple inputs)
        args.onnx,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=[
            "input_roi_image",
            "input_bb_with_class",
        ],  # the model's input names
        output_names=[
            "output_predicted_distance",
            # "output_predicted_distance_blackbox",
            # "output_predicted_distance_height_priors",
            # "output_predicted_distance_raytracing",
            # "output_combined_coefficients",
            # "output_predicted_distance_raytracing_classical",
        ],  # the model's output names
        dynamic_axes={
            "input_roi_image": {0: "batch_size"},  # variable lenght axes
            "input_bb_with_class": {0: "batch_size"},  # variable lenght axes
            "output_predicted_distance": {0: "batch_size"},
            # "output_predicted_distance_blackbox": {0: "batch_size"},
            # "output_predicted_distance_height_priors": {0: "batch_size"},
            # "output_predicted_distance_raytracing": {0: "batch_size"},
            # "output_combined_coefficients": {0: "batch_size"},
            # "output_predicted_distance_raytracing_classical": {0: "batch_size"},
        },
    )
