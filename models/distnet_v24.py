import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
import torchmetrics
import torchvision.models as models


# Use dataloader_v6
# for dearsan
class DistNet_v24(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super(DistNet_v24, self).__init__()

        self.lr = learning_rate
        # IMAGE_SIZE_X = 2880
        # IMAGE_SIZE_Y = 1860
        # CAMERA_FOV = 53.8825689 # horizontal
        # CAMERA_HEIGHT = 3.06
        # FOCAL_LENGTH = 2833.33333333

        # Camera parameters
        self.focal_length = nn.Parameter(
            torch.tensor(2833.33333333), requires_grad=False
        )
        self.c_x = nn.Parameter(torch.tensor(2880 / 2), requires_grad=False)
        self.c_y = nn.Parameter(torch.tensor(1860 / 2), requires_grad=False)

        # Height priors
        self.height_priors = nn.Parameter(
            torch.tensor(
                [
                    3.60,  # 0: ida - Internet: albatros-t 3.8, albatros-s 2.5
                    3.75,  # 1: boat - CARLA
                    12.4,  # 2: ship
                    0.2,  # 3: floatsam
                    20.23,  # 4: sailing - CARLA
                    4.2,  # 5: other
                ]
            ),
            requires_grad=False,
        )

        # Initial plane normal estimate aX + bY + cZ + d = 0
        # self.camera_height_above_water = 2.0
        self.camera_height_above_water = 3.06
        self.plane_normal_a = 0.0
        self.plane_normal_b = -1.0
        self.plane_normal_c = 0.0

        fov_x = 53.8825689 * np.pi / 180.0  # radians
        self.image_width = 2880
        self.z_to_distance_coefficient = 2 / self.image_width * np.tan(fov_x / 2)

        # ROI stream backbone (ResNet18)
        self.roi_backbone = models.regnet_y_400mf(weights="DEFAULT")
        in_features_roi = self.roi_backbone.fc.in_features
        self.backbone_output_size = 128
        self.roi_backbone.fc = nn.Sequential(
            nn.Linear(in_features_roi, 256),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.LayerNorm(128),
            nn.Linear(128, self.backbone_output_size),
        )

        # Bounding box stream (fully connected layers)
        self.bbox_stream_output_size = 32
        self.bbox_input_size = 5
        self.bbox_stream = nn.Sequential(
            nn.Linear(self.bbox_input_size, 128, bias=False),
            nn.LayerNorm(128),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(64, self.bbox_stream_output_size),  # Output for regression
        )

        # Fully connected layers for combining streams and regression
        self.fc_combined = nn.Sequential(
            nn.Linear(
                self.backbone_output_size
                + self.bbox_stream_output_size
                + self.bbox_input_size,
                256,
                bias=False,
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 128, bias=False),
            nn.LayerNorm(128),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(
                128, 2
            ),  # Two outputs: one for distance prediction, one for the coefficient scalar
            nn.Sigmoid(),  # Output for regression (normalized) and coefficients (0 to 1)
        )

        # Fully connected layers for height deviation estimation
        self.fc_height = nn.Sequential(
            nn.Linear(
                self.backbone_output_size
                + self.bbox_input_size
                + 1,  # for height priors[class id]
                512,
                bias=False,
            ),
            nn.LayerNorm(512),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm(256),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 128, bias=False),
            nn.LayerNorm(128),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Sigmoid(),  # Output for regression (normalized) and coefficients (0 to 1)
        )

        self.fc_height_bias = nn.Sequential(
            nn.Linear(
                self.backbone_output_size
                + self.bbox_input_size
                + 1,  # for height priors[class id]
                128,
                bias=False,
            ),
            nn.LayerNorm(128),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.fc_raytracing = nn.Sequential(
            nn.Linear(
                self.backbone_output_size + self.bbox_input_size + 1, 512, bias=False
            ),
            nn.LayerNorm(512),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(512, 256, bias=False),
            nn.LayerNorm(256),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(256, 128, bias=False),
            nn.LayerNorm(128),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(128, 2),  # bias, confidence
            nn.Sigmoid(),
        )

    def forward(self, roi, bounding_box_with_class):
        # Extract features from the ROI stream backbone
        features_roi = self.roi_backbone(roi)

        # Process the bounding_box_with_class information
        processed_bbox = self.bbox_stream(bounding_box_with_class)

        # BLACK BOX ESTIMATOR
        # Concatenate the features from both streams and bounding box information
        combined_features_blackbox = torch.cat(
            [features_roi, processed_bbox, bounding_box_with_class],
            dim=1,
        )

        # Pass through fully connected layers for regression
        predicted_distance_blackbox, coefficient_blackbox = torch.chunk(
            self.fc_combined(combined_features_blackbox), 2, dim=1
        )
        predicted_distance_blackbox = predicted_distance_blackbox * 1267.0

        # ESTIMATOR USING HEIGHT PRIORS
        # Height deviation estimation
        # Get the corresponding height priors for the class ids in the batch
        class_ids = bounding_box_with_class[
            :, 0
        ].long()  # Convert class ids to long integers
        height_prior_for_this_class = torch.index_select(
            self.height_priors, 0, class_ids
        ).unsqueeze(-1)

        combined_features_height_priors = torch.cat(
            [
                features_roi,
                bounding_box_with_class,
                height_prior_for_this_class,
            ],
            dim=1,
        )
        (
            predicted_height_deviation_coefficient,
            coefficient_height_priors,
        ) = torch.chunk(self.fc_height(combined_features_height_priors), 2, dim=1)

        predicted_height_deviation_coefficient = (
            0.2 + 1.9 * predicted_height_deviation_coefficient
        )  # 0.2 - 2.1 of the prior height
        # TODO: I will make this plausible range class-dependent.

        predicted_physical_height = (
            predicted_height_deviation_coefficient * height_prior_for_this_class
        )

        predicted_distance_height_priors_before_bias = (
            self.focal_length
            * predicted_physical_height
            / bounding_box_with_class[:, 4].unsqueeze(-1)
        )

        z_to_distance_coefficient = torch.sqrt(
            1
            + (
                (bounding_box_with_class[:, 1].unsqueeze(-1) - self.image_width / 2)
                * self.z_to_distance_coefficient
            )
            ** 2
        )
        predicted_distance_height_priors_before_bias = (
            predicted_distance_height_priors_before_bias * z_to_distance_coefficient
        )

        combined_features_height_priors_bias = torch.cat(
            [
                features_roi,
                bounding_box_with_class,
                predicted_distance_height_priors_before_bias,
            ],
            dim=1,
        )

        predicted_distance_height_priors = nn.ReLU()(
            predicted_distance_height_priors_before_bias
            + self.fc_height_bias(combined_features_height_priors_bias) * 70.0
        )

        predicted_z_raytracing_classical = (
            -self.camera_height_above_water
            * self.focal_length
            / (
                self.plane_normal_a
                * (bounding_box_with_class[:, 1].unsqueeze(-1) - self.c_x)
                + self.plane_normal_b
                * (bounding_box_with_class[:, 2].unsqueeze(-1) - self.c_y)
                + self.plane_normal_c * self.focal_length
            )
        )
        predicted_distance_classical_raytracing_before_clipping = (
            z_to_distance_coefficient * predicted_z_raytracing_classical
        )
        for_clipping_mask = torch.ones_like(
            predicted_distance_classical_raytracing_before_clipping
        )

        predicted_distance_raytracing_classical = torch.min(
            1267.0 * for_clipping_mask,
            torch.max(
                predicted_distance_classical_raytracing_before_clipping,
                0.0 * for_clipping_mask,
            ),
        )

        combined_features_raytracing = torch.cat(
            [
                features_roi,
                bounding_box_with_class,
                predicted_distance_raytracing_classical,
            ],
            dim=1,
        )

        (
            predicted_raytracing_bias,
            coefficient_raytracing,
        ) = torch.chunk(self.fc_raytracing(combined_features_raytracing), 2, dim=1)

        predicted_raytracing_bias = (predicted_raytracing_bias - 0.5) * 100.0

        predicted_distance_raytracing = (
            predicted_distance_raytracing_classical + predicted_raytracing_bias
        )

        # Combine coefficients using softmax
        combined_coefficients = nn.functional.softmax(
            torch.stack(
                [
                    coefficient_blackbox,
                    coefficient_height_priors,
                    coefficient_raytracing,
                ],
                dim=1,
            ),
            dim=1,
        )

        # Total predicted distance: Convex combination of the estimators
        predicted_distance = (
            combined_coefficients[:, 0] * predicted_distance_blackbox
            + combined_coefficients[:, 1] * predicted_distance_height_priors
            + combined_coefficients[:, 2] * predicted_distance_raytracing
        )

        return (
            predicted_distance,
            # predicted_distance_blackbox,
            # predicted_distance_height_priors,
            # predicted_distance_raytracing,
            # combined_coefficients,
            # predicted_distance_raytracing_classical,
        )

    def training_step(self, batch, batch_idx):
        roi, bounding_box_with_class, target_distance = batch
        (
            predicted_distance,
            predicted_distance_blackbox,
            predicted_distance_height_priors,
            predicted_distance_raytracing,
            combined_coefficients,
            predicted_distance_raytracing_classical,
        ) = self(roi, bounding_box_with_class)
        # print("target_distance", target_distance)
        final_loss = nn.MSELoss()(predicted_distance, target_distance)
        black_box_loss = nn.MSELoss()(predicted_distance_blackbox, target_distance)
        height_prior_loss = nn.MSELoss()(
            predicted_distance_height_priors, target_distance
        )
        raytracing_loss = nn.MSELoss()(predicted_distance_raytracing, target_distance)
        smoothing_loss = nn.MSELoss()(
            combined_coefficients, torch.zeros_like(combined_coefficients)
        )
        loss = (
            final_loss
            + black_box_loss
            + height_prior_loss
            + raytracing_loss
            + smoothing_loss * 2.0
        )
        self.log("train_loss", loss)
        self.log("train_loss_blackbox", black_box_loss)
        self.log("train_loss_height_prior", height_prior_loss)
        self.log("train_loss_raytracing", raytracing_loss)
        self.log("train_loss_smoothing", smoothing_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        roi, bounding_box_with_class, target_distance = batch
        (
            predicted_distance,
            predicted_distance_blackbox,
            predicted_distance_height_priors,
            predicted_distance_raytracing,
            combined_coefficients,
            predicted_distance_raytracing_classical,
        ) = self(roi, bounding_box_with_class)
        val_loss = nn.MSELoss()(predicted_distance, target_distance)
        self.log("val_loss", val_loss)
        val_loss_l1 = nn.L1Loss()(predicted_distance, target_distance)
        self.log("val_loss_l1", val_loss_l1)
        black_box_loss = nn.MSELoss()(predicted_distance_blackbox, target_distance)
        self.log("val_loss_blackbox", black_box_loss)
        black_box_loss = nn.L1Loss()(predicted_distance_blackbox, target_distance)
        self.log("val_loss_blackbox_l1", black_box_loss)
        height_prior_loss = nn.MSELoss()(
            predicted_distance_height_priors, target_distance
        )
        self.log("val_loss_heightprior", height_prior_loss)
        height_prior_loss = nn.L1Loss()(
            predicted_distance_height_priors, target_distance
        )
        self.log("val_loss_heightprior_l1", height_prior_loss)
        raytracing_loss = nn.MSELoss()(predicted_distance_raytracing, target_distance)
        self.log("val_loss_raytracing", raytracing_loss)
        raytracing_loss = nn.L1Loss()(predicted_distance_raytracing, target_distance)
        self.log("val_loss_raytracing_l1", raytracing_loss)

        classical_raytracing_loss = nn.L1Loss()(
            predicted_distance_raytracing_classical, target_distance
        )
        self.log("val_loss_classical_raytracing_l1", classical_raytracing_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
