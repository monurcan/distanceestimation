import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning
import torch
import torch.utils.data
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class YOLOLikeDataset(Dataset):
    def __init__(self, image_dir, label_dir, distance_dir, yolo_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.distance_dir = distance_dir
        self.yolo_dir = yolo_dir
        # self.feature_dir = feature_dir
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
        self.annotation_names = natsorted(
            [
                os.path.relpath(path, self.label_dir)
                for path in self.label_dir.rglob("*.txt")
            ]
        )
        # -1, 0, 1, 2, 3,
        # self.x_mid_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]
        # self.y_mid_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]
        # self.width_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]
        # self.height_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]

        # self.feature_extractor = Extractor(
        #     "/mnt/onurcan/deepsort/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
        # )
        
        with open(os.path.join(self.yolo_dir, "predicted_and_gt_distances.txt")) as yolo_gts:
            self.yolo_gt_distances = yolo_gts.readlines()
            # one line: gt distance, prediction 1, prediction 2
            self.yolo_gt_distances = [x.split(" ")[0] for x in self.yolo_gt_distances]
            self.yolo_gt_distances = list(map(float, self.yolo_gt_distances))
            self.len_yolo_gt_distances = len(self.yolo_gt_distances)

    def __len__(self):
        return len(self.annotation_names)

    @staticmethod
    def generate_random_with_histogram(hist):
        return float(np.random.choice(np.arange(-1, len(hist) - 1), p=hist))
    
    @staticmethod
    def generate_random_scale():
        random_var1 = random.triangular(-0.15, +0.24, 0)
        if abs(random_var1) > 0.09 and random.random() < 0.10:
            return 1.0
        return 1+random_var1

    def __getitem__(self, idx):
        annotation_name = self.annotation_names[idx]
        annotation_path = os.path.join(self.label_dir, annotation_name)
        image_path = os.path.join(
            self.image_dir, annotation_name.replace(".txt", ".png")
        )
        distance_path = os.path.join(self.distance_dir, annotation_name)
        # feature_path = os.path.join(self.feature_dir, annotation_name)
        if (
            not os.path.exists(distance_path)
            or not os.path.exists(image_path)
            # or not os.path.exists(feature_path)
        ):
            # Handle the case where the annotation or image file is missing
            print("Missing file", annotation_name)
            return None

        try:
            image = cv2.imread(image_path)[:,:,::-1]
        except:
            print(image_path)
            return None
        
        WIDTH = 2840
        HEIGHT = 2840
        image_shape = (HEIGHT, WIDTH)

        with open(annotation_path, "r") as f:
            annotations = f.readlines()

        with open(distance_path, "r") as f:
            distances = f.readlines()

        # with open(feature_path, "r") as f:
        #     features = f.readlines()

        # Filter non-empty lines
        valid_annotations = [line.strip() for line in annotations if line.strip()]
        valid_distances = [line.strip() for line in distances if line.strip()]
        # valid_features = [line.strip() for line in features if line.strip()]

        if len(valid_annotations) != len(valid_distances):
            # Handle the case where the number of bounding boxes and distances are not equal
            return None

        valid_lines = list(zip(valid_annotations, valid_distances))

        samples = []
        if idx < self.len_yolo_gt_distances:
            with open(os.path.join(self.yolo_dir, "input_outputs_distance_model", f"bb_info_{idx}.txt")) as f_yolo_bb_info:
                yolo_bb_info = f_yolo_bb_info.readlines()
                yolo_bb_info = [float(x) for x in yolo_bb_info[0].split(" ")]
                yolo_bb_with_class = torch.tensor(yolo_bb_info)
                
            yolo_roi = cv2.imread(os.path.join(self.yolo_dir, "input_outputs_distance_model", f"roi_image_{idx}.png"))[:,:,::-1]
            if self.transform:
                try:
                    yolo_roi = self.transform(yolo_roi)
                except ValueError:
                    print(os.path.join(self.yolo_dir, "input_outputs_distance_model", f"roi_image_{idx}.png"))
                    print(yolo_roi.shape)
                    return None
                
            yolo_target_distance = torch.tensor([self.yolo_gt_distances[idx]])
            samples.append(
                [
                    yolo_roi,
                    yolo_bb_with_class,
                    yolo_target_distance,
                ]
            )
            
        if not valid_lines:
            # Handle the case where all lines are empty
            return samples
        
        for selected_line_i in range(len(valid_annotations)):
            # Randomly select one bounding box
            # selected_line, selected_distance, selected_feature = random.choice(valid_lines)
            selected_line, selected_distance = valid_lines[selected_line_i]

            ann_parts = selected_line.split()
            class_id, x_center, y_center, width, height = map(float, ann_parts)
            x_center *= image_shape[1]
            y_center *= image_shape[0]
            width *= image_shape[1]
            height *= image_shape[0]
            object_distance = float(selected_distance)

            # Augmentation for robustness to small deviations
            # x_mid_deviation = self.generate_random_with_histogram(
            #     self.x_mid_deviation_hist
            # )
            # y_mid_deviation = self.generate_random_with_histogram(
            #     self.y_mid_deviation_hist
            # )
            # width_deviation = self.generate_random_with_histogram(
            #     self.width_deviation_hist
            # )
            # height_deviation = self.generate_random_with_histogram(
            #     self.height_deviation_hist
            # )
            width_deviation = self.generate_random_scale()
            height_deviation = self.generate_random_scale()
            x_mid_deviation = (self.generate_random_scale() - 1)/2 * width
            y_mid_deviation = (self.generate_random_scale() - 1)/2 * height

            x_center, y_center, width, height = (
                x_center + x_mid_deviation,
                y_center + y_mid_deviation,
                width_deviation * width,
                height_deviation * height,
            )

            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            # Handle the case where the bounding box is out of the image
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image_shape[1], x_max)
            y_max = min(image_shape[0], y_max)

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            if width == 0 or height == 0:
                continue

            # Convert the selected bounding box information to tensors
            target_distance = torch.tensor([object_distance])

            # Far objects can be detected as other:
            if object_distance > 90:
                if object_distance < 200:
                    other_prob = 3 / 20 * (object_distance - 100) + 5
                else:
                    other_prob = 1 / 20 * (object_distance - 200) + 20

                if random.random() < other_prob / 100:
                    class_id = 5

            bounding_box_with_class = torch.tensor(
                [
                    class_id,
                    x_center,
                    y_max,
                    width,
                    height,
                ]
            )

            # selected_feature = selected_feature.split()
            # # print(selected_feature)
            # selected_feature = list(map(float, selected_feature))
            # # print("selected_feature", selected_feature)
            # selected_feature_tensor = torch.tensor(selected_feature)

            # Generate the cropped ROI
            x_min = max(0, x_min - 10)
            y_min = max(0, y_min - 10)
            x_max = min(image_shape[1], x_max + 10)
            y_max = min(image_shape[0], y_max + 10)
            try:
                roi = image[y_min:y_max, x_min:x_max]
            except TypeError:
                print(x_min, x_max, y_min, y_max)
                print(image_shape)
                print(image_path)
                return None
            # print("roi", roi.shape)

            if self.transform:
                # image = self.transform(image)
                try:
                    roi = self.transform(roi)
                except ValueError:
                    print(y_min, y_max, x_min, x_max)
                    print(image_shape)
                    print(image_path)
                    print(roi.shape)
                    return None

            samples.append(
                [
                    roi,
                    bounding_box_with_class,
                    target_distance,
                ]
            )
        return samples


class DepthDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, image_dir, label_dir, distance_dir, yolo_dir) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.distance_dir = distance_dir
        self.yolo_dir = yolo_dir
        # self.feature_dir = feature_dir

    def setup(self, stage) -> None:
        raw_dataset = YOLOLikeDataset(
            image_dir=self.image_dir,
            label_dir=self.label_dir,
            distance_dir=self.distance_dir,
            yolo_dir=self.yolo_dir,
            # feature_dir=self.feature_dir,
        )
        train_val_split_index = int(0.8 * len(raw_dataset))
        self.train_dataset = torch.utils.data.Subset(
            raw_dataset, range(train_val_split_index)
        )
        self.val_dataset = torch.utils.data.Subset(
            raw_dataset, range(train_val_split_index, int(len(raw_dataset)))
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=120,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=120,
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        batch = [item for sublist in batch for item in sublist]
        # print(len(batch))
        return torch.utils.data.dataloader.default_collate(batch)
