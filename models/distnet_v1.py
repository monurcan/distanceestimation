import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
import torchmetrics
import torchvision.models as models


class DistNet_v1(pl.LightningModule):
    def __init__(self, pretrained=True):
        super(DistNet_v1, self).__init__()

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
            nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()
        )

        # Fully connected layers for combining streams and regression
        self.fc_combined = nn.Sequential(
            nn.Linear(in_features_image + in_features_roi + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
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
            [features_image, features_roi, processed_bbox], dim=1
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
        return torch.optim.Adam(self.parameters(), lr=1e-3)
