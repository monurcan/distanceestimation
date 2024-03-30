import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
import torchmetrics
import torchvision.models as models


class DistNet_v2(pl.LightningModule):
    def __init__(self, pretrained=True, learning_rate=1e-3):
        super(DistNet_v2, self).__init__()

        self.lr = learning_rate

        # Image stream backbone (ResNet50)
        self.image_backbone = models.resnet50(pretrained=pretrained)
        in_features_image = self.image_backbone.fc.in_features
        self.image_backbone.fc = nn.Identity()

        # ROI stream backbone (ResNet18)
        self.roi_backbone = models.resnet18(pretrained=pretrained)
        in_features_roi = self.roi_backbone.fc.in_features
        self.roi_backbone.fc = nn.Identity()

        # Bounding box stream (fully connected layers)
        self.bbox_stream = nn.Sequential(
            nn.Linear(5, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),  # Output for regression
            nn.Sigmoid(),
        )

        # Fully connected layers for combining streams and regression
        self.fc_combined = nn.Sequential(
            nn.Linear(in_features_image + in_features_roi + 16 + 5, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),  # Output for regression
        )

    def forward(self, image, roi, bounding_box_with_class):
        # Extract features from the image stream backbone
        features_image = self.image_backbone(image)

        # Extract features from the ROI stream backbone
        features_roi = self.roi_backbone(roi)

        # Process the bounding_box_with_class information
        processed_bbox = self.bbox_stream(bounding_box_with_class)

        # Concatenate the features from both streams and bounding box information
        combined_features = torch.cat(
            [features_image, features_roi, processed_bbox, bounding_box_with_class],
            dim=1,
        )

        # Pass through fully connected layers for regression
        predicted_distance = self.fc_combined(combined_features)

        return predicted_distance

    def training_step(self, batch, batch_idx):
        image, roi, bounding_box_with_class, target_distance = batch
        predicted_distance = self(image, roi, bounding_box_with_class)
        loss = nn.MSELoss()(predicted_distance, target_distance)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, roi, bounding_box_with_class, target_distance = batch
        predicted_distance = self(image, roi, bounding_box_with_class)
        val_loss = nn.MSELoss()(predicted_distance, target_distance)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
