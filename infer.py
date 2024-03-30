"""
    Example use:
        python3 infer.py --checkpoint_path "logs/exp_v21_oncekiv20aslindav19mus/version_0/checkpoints/epoch=169-step=22440.ckpt" --image_path "/mnt/onurcan/onurcanDenemTask/images_stabilized" --label_path "/mnt/onurcan/onurcanDenemTask/labels_stabilized" --output_path "logs/exp_v21_oncekiv20aslindav19mus/version_0/inference" --save_video --no_deviation --ext "jpg" --gt_distance_path ""
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import models

sys.path.append("/mnt/onurcan/deepsort/deep_sort_pytorch/deep_sort/deep")
# This is not the best practice, I should change PYTHONPATH variable, but, I like explicitly setting it in the code.
# https://github.com/ZQPei/deep_sort_pytorch
from feature_extractor import Extractor


def generate_random_with_histogram(hist):
    return float(np.random.choice(np.arange(-1, len(hist) - 1), p=hist))


def main(
    label_path,
    gt_distance_path,
    image_path,
    image,
    model_,
    no_deviation: bool = False,
    feature_extractor=None,
):
    # Prepare the bounding_box_with_class
    with open(label_path, "r") as f:
        annotations = f.readlines()
    valid_annotations = [line.strip() for line in annotations if line.strip()]

    if len(valid_annotations) == 0:
        print("No valid annotations found")
        return

    for selected_line_i in range(len(valid_annotations)):
        # selected_line_i = random.randint(0, len(valid_annotations) - 1)

        selected_line = valid_annotations[selected_line_i]
        ann_parts = selected_line.split()
        class_id, x_center, y_center, width, height = map(float, ann_parts)

        # after v4_augmentation_with_deviations
        x_center *= image.shape[1]
        y_center *= image.shape[0]
        width *= image.shape[1]
        height *= image.shape[0]

        if no_deviation:
            x_mid_deviation_hist = [0.0, 1.0, 0.0, 0.0, 0.0]
            y_mid_deviation_hist = [0.0, 1.0, 0.0, 0.0, 0.0]
            width_deviation_hist = [0.0, 1.0, 0.0, 0.0, 0.0]
            height_deviation_hist = [0.0, 1.0, 0.0, 0.0, 0.0]
        else:
            x_mid_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]
            y_mid_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]
            width_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]
            height_deviation_hist = [0.05, 0.55, 0.25, 0.10, 0.05]

        x_mid_deviation = generate_random_with_histogram(x_mid_deviation_hist)
        y_mid_deviation = generate_random_with_histogram(y_mid_deviation_hist)
        width_deviation = generate_random_with_histogram(width_deviation_hist)
        height_deviation = generate_random_with_histogram(height_deviation_hist)

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
            continue

        # since the models are not trained for classes other than boat 1 and sail 4
        # class_id = 4.0 if class_id == 4.0 else 1.0
        # class_id = 1.0 if class_id == 0.0 else class_id
        # class_id = 1.0

        bounding_box_with_class_cpu = [
            class_id,
            x_center,
            y_max,
            width,
            height,
        ]
        bounding_box_with_class = torch.tensor(bounding_box_with_class_cpu).unsqueeze(
            0
        )  # Add batch dimension

        selected_distance = None
        if gt_distance_path:
            with open(gt_distance_path, "r") as f:
                distances = f.readlines()

            valid_distances = [line.strip() for line in distances if line.strip()]
            if len(valid_distances) != len(valid_annotations):
                print("Number of bounding boxes and distances are not equal")
                continue
            selected_distance = float(valid_distances[selected_line_i])

        # Load and preprocess the image and ROI (assuming you have the paths)
        x_min = max(0, x_min - 2)
        y_min = max(0, y_min - 2)
        x_max = min(image.shape[1], x_max + 2)
        y_max = min(image.shape[0], y_max + 2)

        roi = image[y_min:y_max, x_min:x_max]

        features = None
        if feature_extractor:
            features = feature_extractor([roi])

        transform = transforms.Compose(
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

        image_ = transform(image).unsqueeze(0)  # Add batch dimension
        try:
            roi = transform(roi).unsqueeze(0)  # Add batch dimension
        except ValueError:
            print(y_min, y_max, x_min, x_max)
            print(image_.shape)
            print(image_path)
            continue

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_ = image_.to(device)
        roi = roi.to(device)
        bounding_box_with_class = bounding_box_with_class.to(device)

        # Perform inference
        with torch.no_grad():
            # predicted_distance = model_(image_, roi, bounding_box_with_class)
            if features is not None:
                features = torch.tensor(features).to(device)
                model_output = model_(features, bounding_box_with_class)
            else:
                model_output = model_(roi, bounding_box_with_class)

        (
            predicted_distance,
            predicted_distance_blackbox,
            predicted_distance_height_priors,
            predicted_distance_raytracing,
            combined_coefficients,
            predicted_distance_raytracing_classical,
        ) = model_output
        predicted_distance = predicted_distance.item()

        yield (
            selected_distance,
            predicted_distance,
            width_deviation,
            height_deviation,
            x_mid_deviation,
            y_mid_deviation,
            bounding_box_with_class_cpu,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the model checkpoint. Do not forget to change the model class in the script!",
        default="logs/exp_v17/version_0/checkpoints/epoch=34-step=5320.ckpt",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the input image",
        default="/mnt/carla_capture_seq/cam_rgb/",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        help="Path to the BB label",
        default="/mnt/carla_capture_seq/labels/",
    )
    parser.add_argument(
        "--gt_distance_path",
        type=str,
        help="Path to GT distance label",
        default="/mnt/carla_capture_seq/true_distance_not_z/",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the outputs (errors etc.)",
        default="logs/exp_v17/version_0/inference",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Whether to save the images with bounding boxes",
    )
    parser.add_argument(
        "--sorted",
        action="store_true",
        help="Whether to sort the images, our dataloader does not sort the images by default",
    )
    parser.add_argument(
        "--split_index",
        type=float,
        help="Index to split the dataset into train and validation sets",
        default=0.8,
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="FPS of the video",
        default=10,
    )
    parser.add_argument("--ext", type=str, help="Input image extension", default="png")
    parser.add_argument("--no_deviation", action="store_true", default=False)
    parser.add_argument("--smooth", action="store_true", default=False)
    parser.add_argument(
        "--before_dataloaderv4_feature_extract", action="store_false", default=True
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load feature extractor
    feature_extractor = None
    if args.before_dataloaderv4_feature_extract:
        feature_extractor = Extractor(
            "/mnt/onurcan/deepsort/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
        )

    # Load the trained model
    model = models.DistNet_v23.load_from_checkpoint(args.checkpoint_path)

    # Set model to evaluation mode
    model.eval()

    # Load dataset
    annotation_names = [
        os.path.relpath(path, args.label_path)
        for path in Path(args.label_path).rglob("*.txt")
    ]
    annotation_names = sorted(annotation_names) if args.sorted else annotation_names

    train_val_split_index = int(args.split_index * len(annotation_names))
    annotation_names = annotation_names[train_val_split_index:]

    metrics = {
        "predicted_distances": [],
        "gt_distances": [],
        "signed_errors": [],
        "abs_errors": [],
        "l2_errors": [],
        "width_deviation": [],
        "height_deviation": [],
        "x_mid_deviation": [],
        "y_mid_deviation": [],
    }

    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            os.path.join(args.output_path, "video.mp4"), fourcc, args.fps, (1500, 1500)
        )

    i = 0
    while i < len(annotation_names):
        annotation_name = annotation_names[i]
        i += 1

        annotation_path = os.path.join(args.label_path, annotation_name)
        image_path = os.path.join(
            args.image_path, annotation_name.replace(".txt", "." + args.ext)
        )
        image = cv2.imread(image_path)

        distance_path = None
        if args.gt_distance_path:
            distance_path = os.path.join(args.gt_distance_path, annotation_name)

        results_ = main(
            annotation_path,
            distance_path,
            image_path,
            image,
            model,
            args.no_deviation,
            feature_extractor,
        )
        if args.save_video:
            image_visualize = image.copy()
        for results in results_:
            if results:
                (
                    selected_distance,
                    predicted_distance,
                    width_deviation,
                    height_deviation,
                    x_mid_deviation,
                    y_mid_deviation,
                    bounding_box_with_class,
                ) = results
                if not selected_distance:
                    selected_distance = predicted_distance
                print(
                    f"GT distance: {selected_distance:.2f}, Predicted distance: {predicted_distance:.2f}"
                )
                error = selected_distance - predicted_distance
                abs_error = abs(error)

                metrics["predicted_distances"].append(predicted_distance)
                metrics["gt_distances"].append(selected_distance)
                metrics["signed_errors"].append(error)
                metrics["abs_errors"].append(abs_error)
                metrics["l2_errors"].append(abs_error**2)
                metrics["width_deviation"].append(width_deviation)
                metrics["height_deviation"].append(height_deviation)
                metrics["x_mid_deviation"].append(x_mid_deviation)
                metrics["y_mid_deviation"].append(y_mid_deviation)
                print(
                    f"Deviation: {width_deviation:.2f}, {height_deviation:.2f}, {x_mid_deviation:.2f}, {y_mid_deviation:.2f}"
                )
                print(
                    f"Absolute error: {abs_error:.2f}, Absolute error mean: {sum(metrics['abs_errors'])/len(metrics['abs_errors']):.2f}, Absolute l2 error mean: {np.sqrt(sum(metrics['l2_errors'])/len(metrics['l2_errors'])):.2f}\n"
                )

                if args.save_video:
                    (
                        class_id,
                        x_center,
                        y_max,
                        width,
                        height,
                    ) = bounding_box_with_class
                    print("class_id", class_id)
                    x_min = x_center - width / 2
                    y_min = y_max - height
                    cv2.rectangle(
                        image_visualize,
                        (int(x_min), int(y_min)),
                        (int(x_min + width), int(y_min + height)),
                        (0, 255, 0),
                        1,
                    )
                    cv2.putText(
                        image_visualize,
                        f"{predicted_distance:.2f} m",
                        (int(x_min), int(y_min)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    if args.gt_distance_path:
                        cv2.putText(
                            image_visualize,
                            f"GT distance: {selected_distance:.2f}",
                            (int(x_min), int(y_min + height + 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
        if args.save_video:
            image_visualize = cv2.resize(image_visualize, (1500, 1500))
            video.write(image_visualize)
            # cv2.imshow("Visualizer", image_visualize)
            # key = cv2.waitKey(0)
            # if key == ord("q"):
            #     break
            # elif key == ord("d"):
            #     i -= 2

    # cv2.destroyAllWindows()
    if args.save_video:
        video.release()
        print(f"Video saved to {os.path.join(args.output_path, 'video.mp4')}")

    # Save the metrics
    with open(os.path.join(args.output_path, "errors.pkl"), "wb") as f:
        pickle.dump(metrics, f)
        print(f"Errors saved to {os.path.join(args.output_path, 'errors.pkl')}")

    for key, value in metrics.items():
        metrics[key] = np.array(value)

    number_of_columns = 5
    number_of_rows = 2 + 5

    # Create a 1x2 grid of plots
    plt.figure(figsize=(5 * number_of_columns, 5 * number_of_rows))
    plt.subplot(
        number_of_rows, number_of_columns, 1
    )  # First subplot for Predicted vs. GT distance
    plt.plot(metrics["gt_distances"], metrics["predicted_distances"], "o")
    plt.plot(metrics["gt_distances"], metrics["gt_distances"], "--")
    plt.title("Predicted vs. GT distance")
    plt.xlabel("GT distance (m)")
    plt.ylabel("Predicted distance (m)")
    plt.grid(which="both")
    plt.minorticks_on()
    plt.tick_params(which="minor", bottom=False, left=False)

    # Calculate percentage error
    percentage_error = (
        100
        * (metrics["predicted_distances"] - metrics["gt_distances"])
        / metrics["gt_distances"]
    )

    plt.subplot(
        number_of_rows, number_of_columns, 2
    )  # Second subplot for Percentage Error
    plt.plot(metrics["gt_distances"], percentage_error, "o")
    plt.axhline(
        y=0, color="r", linestyle="--"
    )  # Add a horizontal line at y=0 for reference
    plt.title("Percentage Error vs. GT distance")
    plt.xlabel("GT distance (m)")
    plt.ylabel("Percentage Error (%)")
    plt.grid(which="both")
    plt.minorticks_on()
    plt.tick_params(which="minor", bottom=False, left=False)

    plt.subplot(number_of_rows, number_of_columns, 3)
    ypbot = np.percentile(percentage_error, 1)
    yptop = np.percentile(percentage_error, 99)
    ypad = 0.2 * (yptop - ypbot)
    ymin = ypbot - ypad
    ymax = yptop + ypad

    plt.plot(metrics["gt_distances"], percentage_error, "o")
    plt.axhline(
        y=0, color="r", linestyle="--"
    )  # Add a horizontal line at y=0 for reference
    plt.title("Percentage Error vs. GT distance (no outliers)")
    plt.xlabel("GT distance (m)")
    plt.ylabel("Percentage Error (%)")
    plt.grid(which="both")
    plt.minorticks_on()
    plt.tick_params(which="minor", bottom=False, left=False)
    plt.ylim([ymin, ymax])

    plt.subplot(number_of_rows, number_of_columns, 4)
    plt.hist(metrics["abs_errors"], bins=50, edgecolor="black")
    plt.title("Histogram of Absolute Errors")
    plt.xlabel("Absolute Error (m)")
    plt.ylabel("Frequency")

    plt.subplot(number_of_rows, number_of_columns, 5)
    metrics_txt = f"""Total samples: {len(metrics['abs_errors'])}

    Average absolute error: {np.mean(metrics['abs_errors'])} (m)
    Average l2 error (RMSE): {np.sqrt(np.mean(metrics['l2_errors']))} (m)
    Average abs. percentage error: {np.mean(np.abs(percentage_error))} (%)

    Standard deviation of absolute error: {np.std(metrics['abs_errors'])} (m)
    Standard deviation of l2 error: {np.std(metrics['l2_errors'])} (m)
    Standard deviation of abs. percentage error: {np.std(np.abs(percentage_error))} (%)
    """

    plt.text(0.5, 0.5, metrics_txt, fontsize=12, ha="center", va="center")
    plt.axis("off")  # Turn off axes for this subplot

    deviation_txt = "For one deviation plot, the other deviations may not be 0.\n\n"
    for i, deviation_type in enumerate(
        ["width_deviation", "height_deviation", "x_mid_deviation", "y_mid_deviation"]
    ):
        deviation_txt += f"Average absolute/percentage error for {deviation_type}:\n"
        deviation_txt_abs = ""
        deviation_txt_percentage = ""
        for deviation_i, deviation in enumerate([-1, 0, 1, 2, 3]):
            predicted_distances_deviated = metrics["predicted_distances"][
                metrics[deviation_type] == deviation
            ]
            gt_distances_deviated = metrics["gt_distances"][
                metrics[deviation_type] == deviation
            ]
            plt.subplot(
                number_of_rows, number_of_columns, number_of_columns * 1 + 1 + i
            )
            plt.plot(
                gt_distances_deviated, predicted_distances_deviated, "o", alpha=0.2
            )

            deviation_txt_abs += f"{deviation}: {np.mean(metrics['abs_errors'][metrics[deviation_type] == deviation]):.2f}, "
            deviation_txt_percentage += f"{deviation}: {np.mean(np.abs(percentage_error[metrics[deviation_type] == deviation])):.2f}, "

            plt.subplot(
                number_of_rows,
                number_of_columns,
                number_of_columns * 2 + 1 + i + deviation_i * number_of_columns,
            )
            plt.plot(gt_distances_deviated, predicted_distances_deviated, "o")

        deviation_txt += f"{deviation_txt_abs}(m)\n{deviation_txt_percentage}(%)\n\n"
        plt.subplot(number_of_rows, number_of_columns, 6 + i)
        plt.plot(metrics["gt_distances"], metrics["gt_distances"], "--")
        plt.title(f"Predicted vs. GT distance\nfor different {deviation_type}")
        plt.xlabel("GT distance (m)")
        plt.ylabel("Predicted distance (m)")
        plt.legend(["-1", "0", "1", "2", "3"])
        plt.grid(which="both")
        plt.minorticks_on()
        plt.tick_params(which="minor", bottom=False, left=False)
        for deviation_i, deviation in enumerate([-1, 0, 1, 2, 3]):
            plt.subplot(
                number_of_rows,
                number_of_columns,
                number_of_columns * 2 + 1 + i + deviation_i * number_of_columns,
            )
            plt.plot(metrics["gt_distances"], metrics["gt_distances"], "--")
            plt.title(f"Predicted vs. GT distance\nfor {deviation_type} = {deviation}")
            plt.xlabel("GT distance (m)")
            plt.ylabel("Predicted distance (m)")
            plt.grid(which="both")
            plt.minorticks_on()
            plt.tick_params(which="minor", bottom=False, left=False)

    plt.subplot(number_of_rows, number_of_columns, 10)
    plt.text(0.5, 0.5, deviation_txt, fontsize=12, ha="center", va="center")
    plt.axis("off")  # Turn off axes for this subplot

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.savefig(os.path.join(args.output_path, "metrics.png"))
    print(f"Metrics saved to {os.path.join(args.output_path, 'metrics.png')}")
    plt.close()
