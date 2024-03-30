"""
This script is used to parse the data from the CARLA simulator
and generate the dataset for training the neural network. The dataset
consists of images, YOLO bounding boxes and the distance of the object from the
camera.

Example usage:
    python3 carla_parser.py --dataset_path "/mnt/Bedo_CARLA/carla-capture-5/" --feature_extractor_checkpoint "/mnt/onurcan/deepsort/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
    
    
Change FOV accordingly!
"""


import argparse
import functools
import sys
from pathlib import Path

sys.path.append("/mnt/onurcan/deepsort/deep_sort_pytorch/deep_sort/deep")
import cv2
import matplotlib.pyplot as plt
import numpy as np

# This is not the best practice, I should change PYTHONPATH variable, but, I like explicitly setting it in the code.
# https://github.com/ZQPei/deep_sort_pytorch
from feature_extractor import Extractor
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map

IDA_LABEL = "0"
BOAT_LABEL = "1"
SHIP_LABEL = "2"
FLOATSAM_LABEL = "3"
SAILING_LABEL = "4"
OTHERS_LABEL = "5"


def mask_to_polygon(mask: np.array, report: bool = False):
    # for sailings (sometimes fragmented)
    kernel = np.ones((6, 6), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    AREA_THRES = 45
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    areas = []
    for object in contours:
        areas.append(cv2.contourArea(object))

        if cv2.contourArea(object) < AREA_THRES:
            continue

        if len(object) > 140:
            epsilon = 0.002 * cv2.arcLength(object, True)
            object = cv2.approxPolyDP(object, epsilon, True)

        coords = []

        for point in object:
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))
        polygons.append(coords)
    # print(sorted(areas))
    if report:
        print(f"Number of points = {len(polygons[0])}")

    return np.array(polygons).tolist()


def depth_to_distance(depth_map: np.array):
    """
    This function takes in the depth map and returns the distance of the object
    from the camera.
    """
    depth_map = depth_map.astype(np.float32)
    B, G, R = cv2.split(depth_map)

    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized
    return in_meters


def process_image(im_pth, cam_fov_x, log_grayscale_depth=True, feature_extractor=None):
    # print(im_pth)
    mask = cv2.imread(str(im_pth))
    if mask is None:
        return

    mask = mask[:, :, 2] + mask[:, :, 1] + mask[:, :, 0] * 3
    mask = mask.astype(np.uint8)  # overflow can happen after summation
    # print(mask)
    HEIGHT, WIDTH = mask.shape

    depth_path_prefix = "/raw/" if log_grayscale_depth else "/"
    depth_path = (
        str(im_pth.parent).replace("cam_segment", "cam_depth")
        + depth_path_prefix
        + im_pth.name
    )
    depth_map = cv2.imread(depth_path)
    if depth_map is None:
        return

    distance_map = depth_to_distance(depth_map)

    if feature_extractor:
        rgb_path = (
            str(im_pth.parent).replace("cam_segment", "cam_rgb")
            + depth_path_prefix
            + im_pth.name
        )
        rgb_image = cv2.imread(rgb_path)

    # fov_x = 75.77 * np.pi / 180.0  # radians
    # fov_x = 53.8825689 * np.pi / 180.0  # radians
    fov_x = cam_fov_x * np.pi / 180.0  # radians

    z_to_distance_coefficient = 2 / WIDTH * np.tan(fov_x / 2)

    ground_sky_water_region = np.isin(mask, [0, 228, 68, 43])
    mask[ground_sky_water_region] = 0

    different_labels = np.unique(mask)

    polygons_boats = []
    polygons_sails = []
    labels_sails = [90, 95, 100, 105, 110, 115, 120, 125]
    polygons_buoys = []
    labels_buoys = [
        180,
        185,
        190,
        195,
        200,
        205,
        210,
        215,
        220,
        225,
        230,
        235,
        240,
        245,
    ]
    polygons_ida = []  # 55 - Pneumatic02?
    labels_ida = [55]
    polygons_big_ships = []
    labels_big_ships = [130, 135, 140, 145, 150, 155, 160, 165, 170, 175]

    for label in different_labels:
        if label == 0:
            continue

        region = mask == label
        if label in labels_sails:
            polygons_sails.extend(
                mask_to_polygon(region.astype(np.uint8), report=False)
            )
        elif label in labels_buoys:
            polygons_buoys.extend(
                mask_to_polygon(region.astype(np.uint8), report=False)
            )
        elif label in labels_ida:
            polygons_ida.extend(mask_to_polygon(region.astype(np.uint8), report=False))
        elif label in labels_big_ships:
            polygons_big_ships.extend(
                mask_to_polygon(region.astype(np.uint8), report=False)
            )
        else:
            polygons_boats.extend(
                mask_to_polygon(region.astype(np.uint8), report=False)
            )

    txt_pth = str(im_pth).replace("cam_segment", "labels_seg").replace("png", "txt")
    txt_pth_od = str(im_pth).replace("cam_segment", "labels").replace("png", "txt")
    txt_pth_distance = (
        str(im_pth).replace("cam_segment", "distance").replace("png", "txt")
    )
    txt_pth_true_distance = (
        str(im_pth).replace("cam_segment", "true_distance_not_z").replace("png", "txt")
    )
    txt_pth_features = (
        str(im_pth).replace("cam_segment", "features").replace("png", "txt")
    )

    if feature_extractor:
        f_features = open(txt_pth_features, "w")

    with open(txt_pth, "w") as f, open(txt_pth_od, "w") as f_od, open(
        txt_pth_distance, "w"
    ) as f_dist, open(txt_pth_true_distance, "w") as f_true_distance:
        for polygon_list, label in [
            (polygons_boats, BOAT_LABEL),
            (polygons_buoys, FLOATSAM_LABEL),
            (polygons_ida, IDA_LABEL),
            (polygons_big_ships, SHIP_LABEL),
            (polygons_sails, SAILING_LABEL),
        ]:
            for polygon in polygon_list:
                polygon[0::2] = [x / WIDTH for x in polygon[0::2]]
                polygon[1::2] = [x / HEIGHT for x in polygon[1::2]]

                # Instance segmentation labels (YOLO format)
                instance_txt = f"{label} {' '.join(str(x) for x in polygon)}\n"

                # 2D BB labels (YOLO format)
                max_x = max(polygon[0::2])
                max_y = max(polygon[1::2])
                max_y_index = polygon.index(max_y)
                min_x = min(polygon[0::2])
                min_y = min(polygon[1::2])

                width = max_x - min_x
                height = max_y - min_y
                center_x = min_x + (width / 2)
                center_y = min_y + (height / 2)
                od_txt = f"{label} {center_x} {center_y} {width} {height}\n"

                # Distance labels
                max_y_for_distance = int(max_y * HEIGHT)
                x_corresponding_to_max_y = polygon[max_y_index - 1]
                x_corresponding_to_max_y = int(x_corresponding_to_max_y * WIDTH)
                z_buffer = distance_map[max_y_for_distance, x_corresponding_to_max_y]
                # but if there is another object in front of it, will it be the distance of that object?
                if z_buffer > 999.9:
                    continue

                f.write(instance_txt)
                f_od.write(od_txt)
                f_dist.write(f"{z_buffer}\n")

                # True distance labels (not z-buffer)
                true_distance = z_buffer * np.sqrt(
                    1
                    + (
                        (x_corresponding_to_max_y - WIDTH / 2)
                        * z_to_distance_coefficient
                    )
                    ** 2
                )
                f_true_distance.write(f"{true_distance}\n")

                # Feature extraction
                if feature_extractor:
                    # print("ONURCAN DEBUGGG", rgb_image.shape)
                    min_x, max_x, min_y, max_y = (
                        int(min_x * WIDTH),
                        int(max_x * WIDTH),
                        int(min_y * HEIGHT),
                        int(max_y * HEIGHT),
                    )
                    try:
                        cropped_img = rgb_image[min_y:max_y, min_x:max_x, :]
                        # print("ONURCAN DEBUG", cropped_img.shape)
                        bb_feature = feature_extractor([cropped_img])
                        # print(bb_feature.shape)
                        bb_feature_str = map(str, bb_feature[0])

                        f_features.write(f"{' '.join(bb_feature_str)}\n")
                    except TypeError:
                        print("TypeError", im_pth)

    if feature_extractor:
        f_features.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA Data Parser")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path of the dataset",
        default="/mnt/carla_capture/",
    )
    parser.add_argument(
        "--log_grayscale_depth",
        action="store_true",
        help="Whether is there any log grayscale depth in the dataset (old versions)",
    )
    parser.add_argument(
        "--feature_extractor_checkpoint",
        type=str,
        help="Path of the feature extractor checkpoint, if none, no feature extraction will be performed",
    )
    parser.add_argument(
        "--cam_fov_x",
        type=float,
        help="Camera FOV in degrees dearsanon 53.8825689, ida 75.77",
        required=True,
    )
    args = parser.parse_args()
    dataset_root = Path(args.dataset_path)

    # Get the list of segmentation masks
    im_pth_list = []
    im_pth_list.extend(sorted((dataset_root / "cam_segment").rglob("*.png")))

    # Create the folders for the labels
    for new_label_folder in [
        "labels_seg",
        "labels",
        "distance",
        "true_distance_not_z",
        "features",
    ]:
        new_label_folder_path = dataset_root / new_label_folder
        new_label_folder_path.mkdir(exist_ok=True)

        for direction in ["left", "right", "mid"]:
            Path(new_label_folder_path / direction).mkdir(exist_ok=True)

    # Feature extractor
    feature_extractor = None
    if args.feature_extractor_checkpoint:
        # Multiprocessing with GPU support is harder, I did not want to deal with it :')

        def process_chunk(chunk, args):
            feature_extractor = Extractor(args.feature_extractor_checkpoint)
            for im_pth in chunk:
                process_image(
                    im_pth,
                    log_grayscale_depth=args.log_grayscale_depth,
                    feature_extractor=feature_extractor,
                    cam_fov_x=args.cam_fov_x,
                )

        num_chunks = 16
        chunk_size = len(im_pth_list) // num_chunks
        chunks = [
            im_pth_list[i : i + chunk_size]
            for i in range(0, len(im_pth_list), chunk_size)
        ]

        process_chunk_partial = functools.partial(process_chunk, args=args)
        process_map(process_chunk_partial, chunks)
    else:
        process_map(
            functools.partial(
                process_image,
                log_grayscale_depth=args.log_grayscale_depth,
                feature_extractor=feature_extractor,
                cam_fov_x=args.cam_fov_x,
            ),
            im_pth_list,
            max_workers=32,
            chunksize=32,
        )
