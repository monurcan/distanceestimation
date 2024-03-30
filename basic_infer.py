import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as lightning
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
import torch
from torchvision import transforms

import models
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distance Estimator ONNX Export")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path of the dataset",
        default="/raid/onurcan/input_outputs_distance_model_fp32/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path of the checkpoint",
        default="/mnt/onurcan/distance_estimation_deep_learning/logs/exp_v23_new_dataset_less_rollpitch_rgb_bgr_order_corrected/version_1/checkpoints/epoch=89-step=6300.ckpt",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    dataset_root = Path(args.dataset_path)
    model = models.DistNet_v23.load_from_checkpoint(args.checkpoint)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # datamodule = dataloader_v6.DepthDataModule(
    #     dataset_root / "cam_rgb",
    #     dataset_root / "labels",
    #     dataset_root / "true_distance_not_z",
    # )
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

    # datamodule.setup("val")
    # roi, bb_with_class, target_distance = next(iter(datamodule.train_dataloader()))
    # roi = torch.randn((342, 3, 224, 224))
    # bb_with_class = torch.tensor([1.0, 232.0, 232.5, 12.22, 32.2]).repeat(342, 1)
    # target_distance = torch.randn((342, 1))
    errors = np.array([])
    percentage_errors = np.array([])

    for i in range(1, 1400):
        with open(dataset_root / f"bb_info_{i}.txt") as f:
            bb_info = f.readlines()
        bb_info = [float(x) for x in bb_info[0].split(" ")]
        bb_with_class = torch.tensor(bb_info).unsqueeze(0)

        roi_image = cv2.imread(str(dataset_root / f"roi_image_{i}.png"))[:, :, ::-1]
        roi = transform(roi_image).unsqueeze(0)

        target_distance = torch.randn((1, 1))

        roi, bb_with_class, target_distance = (
            roi.cuda(),  # .half(),
            bb_with_class.cuda(),  # .half(),
            target_distance.cuda(),  # .half(),
        )
        model.eval()

        # model.half()

        # print(roi.shape, bb_with_class.shape, target_distance.shape)

        # import time
        for _ in range(1):
            # start = time.perf_counter()

            (
                predicted_distance,
                predicted_distance_blackbox,
                predicted_distance_height_priors,
                predicted_distance_raytracing,
                combined_coefficients,
                predicted_distance_raytracing_classical,
            ) = model(roi, bb_with_class)

        # end = time.perf_counter()
        # print(f"%{(end-start)*1000} ms")

        # print(predicted_distance.shape)
        predicted_distance_cpu = predicted_distance.cpu().detach().numpy()[0]
        print(predicted_distance_cpu)

        with open(dataset_root / f"output_{i}.txt") as output:
            output = output.readlines()
        output = [float(x) for x in output[0].split(" ")]
        # print(output.shape)

        print(output)

        error = output[0] - predicted_distance_cpu[0]
        percentage_error = error / predicted_distance_cpu[0] * 100
        print("error:", error)
        print("error percentage:", percentage_error)
        errors = np.append(errors, error)
        percentage_errors = np.append(percentage_errors, percentage_error)
        print("================================================")

    print("mean error:", np.mean(errors))
    print("mean percentage error:", np.mean(percentage_errors))
    print("max percentage error:", np.max(percentage_errors))
    print("max error", np.max(errors))
