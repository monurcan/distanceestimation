import time
from pathlib import Path

import cv2
import numpy as np
from ewasr_segmentation import eWaSR_Segmentation


class ImageStabilizer:
    def __init__(
        self,
        engine_path="/mnt/onurcan/distance_estimation_deep_learning/utils/pretrained_ewasr/ewasr_resnet18.trt",
    ):
        self.engine_path = engine_path
        self.model = eWaSR_Segmentation(engine_path=engine_path)

    def lines_from_segmentation_v2(self, segmentation_image):
        # 0 - obstacle, 1 - sea, 2 - sky
        binary_only_sea = (segmentation_image == 1).astype("uint8")
        binary_only_sky = (segmentation_image == 2).astype("uint8")

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

    def get_stabilized_image(self, original_image):
        result = original_image.copy()

        top_y = 800
        bottom_y = 2350
        slide_y = 5
        slide_x = 5

        segmentation_input = original_image.copy()[
            top_y:bottom_y:slide_y, ::slide_x, :
        ].astype(np.uint8)

        segmentation_output = self.model.infer_single_image(
            segmentation_input,
            (segmentation_input.shape[1], segmentation_input.shape[0]),
        )
        cv2.imshow("segmentation_output", segmentation_output * 100)

        lines = self.lines_from_segmentation_v2(segmentation_output)
        print(lines)
        if lines is not None:
            res_average_lines = self.average_of_lines(lines)
            print(res_average_lines)
            res_average_lines_extended = self.resize_extend(
                res_average_lines, slide_x, slide_y, top_y, original_image.shape[1]
            )

            cv2.line(
                result,
                tuple(res_average_lines_extended[0]),
                tuple(res_average_lines_extended[1]),
                (0, 0, 255),
                3,
            )

            result = self.stabilize_image(result, res_average_lines_extended)

        return result


if __name__ == "__main__":
    stabilizer = ImageStabilizer()

    for path in sorted(
        # Path("/mnt/RawInternalDatasets/USV/SALVO/261222_dearsan_atlas2/").rglob("*.png")
        Path("/mnt/RawInternalDatasets/USV/060323_lucid_ida_saha4/parts/part4").rglob(
            "*.jpg"
        )
    )[3000:]:
        img = cv2.imread(str(path))

        stabilized_image = stabilizer.get_stabilized_image(img)

        stabilized_image_visualize = cv2.resize(stabilized_image, (1000, 1000))
        cv2.imshow("output_data", stabilized_image_visualize)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
