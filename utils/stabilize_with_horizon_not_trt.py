import time
from pathlib import Path

import cv2
import numpy as np


class ImageStabilizer:
    def __init__(self):
        pass

    def lines_from_segmentation_v1(self, segmentation_image):
        binary_only_sea = (segmentation_image == 136).astype("uint8")
        kernel = np.ones((10, 10), np.uint8)
        boundaries = binary_only_sea - cv2.morphologyEx(
            binary_only_sea, cv2.MORPH_ERODE, kernel
        )

        return cv2.HoughLinesP(
            boundaries, 1, np.pi / 180, 80, minLineLength=100, maxLineGap=30
        )

    def lines_from_segmentation_v2(self, segmentation_image):
        # 0 - obstacle, 1 - sea, 2 - sky
        binary_only_sea = (segmentation_image == 136).astype("uint8")
        binary_only_sky = (segmentation_image == 90).astype("uint8")

        boundaries = binary_only_sea - cv2.morphologyEx(
            binary_only_sea, cv2.MORPH_ERODE, np.ones((15, 15), np.uint8)
        )
        boundaries &= cv2.morphologyEx(
            binary_only_sky, cv2.MORPH_DILATE, np.ones((25, 15), np.uint8)
        )
        # cv2.imshow("new boundaries", boundaries*255)

        return cv2.HoughLinesP(
            boundaries, 1, np.pi / 180, 80, minLineLength=130, maxLineGap=30
        )

    def average_of_lines(self, lines):
        points = [(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines]
        mean_points = [int(sum(ele) / len(points)) for ele in zip(*points)]
        return ((mean_points[:2]), (mean_points[2:]))

    def longest_line(self, lines):
        points = [(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines]
        line_lengths = [
            np.sqrt((point[0] - point[2]) ** 2 + (point[1] - point[3]) ** 2)
            for point in points
        ]
        longest_line_i = np.argmax(line_lengths)
        return (
            (lines[longest_line_i][0][0], lines[longest_line_i][0][1]),
            (lines[longest_line_i][0][2], lines[longest_line_i][0][3]),
        )

    def resize_extend(
        self, res_average_lines, slide_x, slide_y, top_y, original_image_W
    ):
        resized_res = (
            (
                res_average_lines[0][0] * slide_x,
                res_average_lines[0][1] * slide_y + top_y,
            ),
            (
                res_average_lines[1][0] * slide_x,
                res_average_lines[1][1] * slide_y + top_y,
            ),
        )
        slope = (resized_res[0][1] - resized_res[1][1]) / (
            resized_res[0][0] - resized_res[1][0]
        )
        return (
            (0, int(resized_res[0][1] - resized_res[0][0] * slope)),
            (
                original_image_W,
                int(resized_res[1][1] + (original_image_W - resized_res[1][0]) * slope),
            ),
        )

    def stabilize_image(
        self, original_image_notchanged, res_average_lines_extended, target_y=1423
    ):
        dx = -(res_average_lines_extended[0][0] - res_average_lines_extended[1][0])
        dy = -(res_average_lines_extended[0][1] - res_average_lines_extended[1][1])
        angle = np.degrees(np.arctan2(dy, dx))

        center = (
            (res_average_lines_extended[0][0] + res_average_lines_extended[1][0]) // 2,
            (res_average_lines_extended[0][1] + res_average_lines_extended[1][1]) // 2,
        )

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        stabilized_image = cv2.warpAffine(
            original_image_notchanged,
            rotation_matrix,
            (original_image_notchanged.shape[1], original_image_notchanged.shape[0]),
        )
        M = np.float32([[1, 0, 0], [0, 1, target_y - center[1]]])
        stabilized_image = cv2.warpAffine(
            stabilized_image,
            M,
            (original_image_notchanged.shape[1], original_image_notchanged.shape[0]),
        )
        return stabilized_image

    def stabilize_point(self, point, res_average_lines_extended, target_y=1423):
        # Calculate the rotation angle
        dx = -(res_average_lines_extended[0][0] - res_average_lines_extended[1][0])
        dy = -(res_average_lines_extended[0][1] - res_average_lines_extended[1][1])
        angle = np.degrees(np.arctan2(dy, dx))

        # Calculate the center point for rotation
        center = (
            (res_average_lines_extended[0][0] + res_average_lines_extended[1][0]) // 2,
            (res_average_lines_extended[0][1] + res_average_lines_extended[1][1]) // 2,
        )

        # Apply rotation to the point
        radian_angle = -np.radians(angle)
        rotated_point = (
            (point[0] - center[0]) * np.cos(radian_angle)
            - (point[1] - center[1]) * np.sin(radian_angle)
            + center[0],
            (point[0] - center[0]) * np.sin(radian_angle)
            + (point[1] - center[1]) * np.cos(radian_angle)
            + center[1],
        )

        # Apply translation to the point
        stabilized_point = (
            int(rotated_point[0]),
            int(rotated_point[1] + (target_y - center[1])),
        )

        return stabilized_point

    def get_rotated_bboxes(
        self,
        image_path,
        res_average_lines_extended,
        original_image_W=2840,
        original_image_H=2840,
    ):
        result = []

        with open(
            image_path.replace("images/", "labels/").replace(".jpg", ".txt")
        ) as f_yolo:
            lines = f_yolo.readlines()
            lines = [line.strip() for line in lines]

        for line in lines:
            bbox = line.split(" ")
            bbox = [float(b) for b in bbox]

            x = bbox[1] * original_image_W
            y = bbox[2] * original_image_H
            w = bbox[3] * original_image_W
            h = bbox[4] * original_image_H

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            x3 = x1
            y3 = y + h / 2
            x4 = x2
            y4 = y - h / 2

            x1_rotated, y1_rotated = self.stabilize_point(
                (x1, y1), res_average_lines_extended
            )
            x2_rotated, y2_rotated = self.stabilize_point(
                (x2, y2), res_average_lines_extended
            )
            x3_rotated, y3_rotated = self.stabilize_point(
                (x3, y3), res_average_lines_extended
            )
            x4_rotated, y4_rotated = self.stabilize_point(
                (x4, y4), res_average_lines_extended
            )

            x_min = min(x1_rotated, x2_rotated, x3_rotated, x4_rotated)
            x_max = max(x1_rotated, x2_rotated, x3_rotated, x4_rotated)
            y_min = min(y1_rotated, y2_rotated, y3_rotated, y4_rotated)
            y_max = max(y1_rotated, y2_rotated, y3_rotated, y4_rotated)

            result.append([int(bbox[0]), x_min, y_min, x_max, y_max])

        return result

    def get_stabilized_image(self, original_image, image_path):
        result = original_image.copy()

        top_y = 800
        bottom_y = 2350
        slide_y = 5
        slide_x = 5

        segmentation_input = original_image.copy()[
            top_y:bottom_y:slide_y, ::slide_x, :
        ].astype(np.uint8)

        # segmentation_output = self.model.infer_single_image(
        #     segmentation_input,
        #     (segmentation_input.shape[1], segmentation_input.shape[0]),
        # )
        segmentation_output = cv2.imread(
            image_path.replace("images/", "images_seg/mask_").replace(".jpg", "m.png")
        )
        segmentation_output = cv2.cvtColor(segmentation_output, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("segmentation_output", segmentation_output * 100)

        lines = self.lines_from_segmentation_v1(segmentation_output)
        print(lines)
        if lines is not None:
            res_average_lines = self.longest_line(lines)
            print(res_average_lines)
            res_average_lines_extended = self.resize_extend(
                res_average_lines, slide_x, slide_y, top_y, original_image.shape[1]
            )

            # cv2.line(
            #     result,
            #     tuple(res_average_lines_extended[0]),
            #     tuple(res_average_lines_extended[1]),
            #     (0, 0, 255),
            #     3,
            # )

            result = self.stabilize_image(
                result,
                res_average_lines_extended,
            )

        rotated_bboxes = self.get_rotated_bboxes(
            image_path,
            res_average_lines_extended=res_average_lines_extended,
            original_image_H=original_image.shape[0],
            original_image_W=original_image.shape[1],
        )

        # for bbox in rotated_bboxes:
        #     cv2.rectangle(
        #         result, (bbox[1], bbox[2]), (bbox[3], bbox[4]), (0, 255, 0), 3
        #     )

        return result, rotated_bboxes


if __name__ == "__main__":
    IDA_LABEL = "0"
    BOAT_LABEL = "1"
    SHIP_LABEL = "2"
    FLOATSAM_LABEL = "3"
    SAILING_LABEL = "4"
    OTHER_LABEL = "5"

    class_mapper = {
        "0": FLOATSAM_LABEL,
        "1": SHIP_LABEL,
        "2": BOAT_LABEL,
        "3": SAILING_LABEL,
        "4": IDA_LABEL,
        "5": OTHER_LABEL,
        "6": IDA_LABEL,
    }

    stabilizer = ImageStabilizer()

    for path in sorted(
        # Path("/mnt/RawInternalDatasets/USV/SALVO/261222_dearsan_atlas2/").rglob("*.png")
        Path("/mnt/onurcan/onurcanDenemeTask2/images").rglob("*.jpg")
    )[0:]:
        img = cv2.imread(str(path))

        stabilized_image, rotated_bboxes = stabilizer.get_stabilized_image(
            img, str(path)
        )

        cv2.imwrite(
            str(path).replace("images/", "images_stabilized/"), stabilized_image
        )

        stabilized_image_visualize = cv2.resize(stabilized_image, (1000, 1000))
        with open(
            str(path).replace("images/", "labels_stabilized/").replace(".jpg", ".txt"),
            "w",
        ) as f_yolo:
            for bbox in rotated_bboxes:
                normalized_bbox = bbox
                normalized_bbox[1] /= stabilized_image.shape[1]
                normalized_bbox[2] /= stabilized_image.shape[0]
                normalized_bbox[3] /= stabilized_image.shape[1]
                normalized_bbox[4] /= stabilized_image.shape[0]
                yolo_bbox = normalized_bbox.copy()
                yolo_bbox[0] = normalized_bbox[0]
                yolo_bbox[1] = (normalized_bbox[1] + normalized_bbox[3]) / 2
                yolo_bbox[2] = (normalized_bbox[2] + normalized_bbox[4]) / 2
                yolo_bbox[3] = normalized_bbox[3] - normalized_bbox[1]
                yolo_bbox[4] = normalized_bbox[4] - normalized_bbox[2]
                yolo_bbox = [str(b) for b in yolo_bbox]
                yolo_bbox[0] = class_mapper[yolo_bbox[0]]
                f_yolo.write(" ".join(yolo_bbox) + "\n")

        # cv2.imshow("output_data", stabilized_image_visualize)

        # if cv2.waitKey(0) & 0xFF == ord("q"):
        #     break

    cv2.destroyAllWindows()
