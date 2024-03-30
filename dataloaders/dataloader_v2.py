import os
import random
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST


class YOLOLikeDataset(Dataset):
    def __init__(self, image_dir, label_dir, distance_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.distance_dir = distance_dir
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.annotation_names = [
            os.path.relpath(path, self.label_dir)
            for path in self.label_dir.rglob("*.txt")
        ]
        # -1, 0, 1, 2, 3,
        self.x_mid_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]
        self.y_mid_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]
        self.width_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]
        self.height_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]

    def __len__(self):
        return len(self.annotation_names)

    @staticmethod
    def generate_random_with_histogram(hist):
        return float(np.random.choice(np.arange(-1, len(hist) - 1), p=hist))

    def __getitem__(self, idx):
        annotation_name = self.annotation_names[idx]
        annotation_path = os.path.join(self.label_dir, annotation_name)
        image_path = os.path.join(
            self.image_dir, annotation_name.replace(".txt", ".png")
        )
        distance_path = os.path.join(self.distance_dir, annotation_name)
        if not os.path.exists(distance_path) or not os.path.exists(image_path):
            # Handle the case where the annotation or image file is missing
            return None

        image = cv2.imread(image_path)

        with open(annotation_path, "r") as f:
            annotations = f.readlines()

        with open(distance_path, "r") as f:
            distances = f.readlines()

        # Filter non-empty lines
        valid_annotations = [line.strip() for line in annotations if line.strip()]
        valid_distances = [line.strip() for line in distances if line.strip()]

        if len(valid_annotations) != len(valid_distances):
            # Handle the case where the number of bounding boxes and distances are not equal
            return None

        valid_lines = list(zip(valid_annotations, valid_distances))

        if not valid_lines:
            # Handle the case where all lines are empty
            return None

        # Randomly select one bounding box
        selected_line, selected_distance = random.choice(valid_lines)
        ann_parts = selected_line.split()
        class_id, x_center, y_center, width, height = map(float, ann_parts)
        x_center *= image.shape[1]
        y_center *= image.shape[0]
        width *= image.shape[1]
        height *= image.shape[0]
        object_distance = float(selected_distance)

        # Augmentation for robustness to small deviations
        x_mid_deviation = self.generate_random_with_histogram(self.x_mid_deviation_hist)
        y_mid_deviation = self.generate_random_with_histogram(self.y_mid_deviation_hist)
        width_deviation = self.generate_random_with_histogram(self.width_deviation_hist)
        height_deviation = self.generate_random_with_histogram(
            self.height_deviation_hist
        )

        x_center, y_center, width, height = (
            x_center + x_mid_deviation,
            y_center + y_mid_deviation,
            width + width_deviation,
            height + height_deviation,
        )

        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Handle the case where the bounding box is out of the image
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        if width == 0 or height == 0:
            return None

        # Convert the selected bounding box information to tensors
        target_distance = torch.tensor([object_distance])
        bounding_box_with_class = torch.tensor(
            [
                class_id,
                x_center,
                y_max,
                width,
                height,
            ]
        )

        # Generate the cropped ROI
        x_min = max(0, x_min - 2)
        y_min = max(0, y_min - 2)
        x_max = min(image.shape[1], x_max + 2)
        y_max = min(image.shape[0], y_max + 2)
        roi = image[y_min:y_max, x_min:x_max]

        if self.transform:
            image = self.transform(image)
            try:
                roi = self.transform(roi)
            except ValueError:
                print(y_min, y_max, x_min, x_max)
                print(image.shape)
                print(image_path)
                return None

        return [image, roi, bounding_box_with_class, target_distance]


class DepthDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, image_dir, label_dir, distance_dir) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.distance_dir = distance_dir

    def setup(self, stage) -> None:
        raw_dataset = YOLOLikeDataset(
            image_dir=self.image_dir,
            label_dir=self.label_dir,
            distance_dir=self.distance_dir,
        )
        train_val_split_index = int(0.8 * len(raw_dataset))
        self.train_dataset = torch.utils.data.Subset(
            raw_dataset, range(train_val_split_index)
        )
        self.val_dataset = torch.utils.data.Subset(
            raw_dataset, range(train_val_split_index, len(raw_dataset))
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=256,
            collate_fn=self.collate_fn,
            num_workers=16,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=256,
            collate_fn=self.collate_fn,
            num_workers=16,
        )

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
