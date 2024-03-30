import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from numpy.lib.function_base import append
from PIL import Image


class TRTInference:
    @staticmethod
    def show_engine_info(engine):
        assert engine is not None

        print("[INFO] TensorRT Engine Info")
        print(f"\t + Max batch size: {engine.max_batch_size}.")
        print(
            f"\t + Engine mem size: {engine.device_memory_size/(1048576)} MB (GPU Mem)."
        )
        print("\t + Tensors:")
        for binding in engine:
            if engine.binding_is_input(binding):
                print(f"\t\t + Input: ", end="")
            else:
                print(f"\t\t + Output: ", end="")
            print(engine.get_binding_shape(binding))

    def __init__(self, engine_path, gpu_num=0, trt_engine_datatype=trt.DataType.FLOAT):
        self.cfx = cuda.Device(gpu_num).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize engine
        with open(engine_path, "rb") as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        self.show_engine_info(engine)
        context = engine.create_execution_context()

        # Get input shape
        dimension = engine.get_binding_shape(engine[0])
        if dimension[1] == 3 or dimension[1] == 1:
            self.channel_first = True
            self.input_channels = dimension[1]
            self.input_height = dimension[2]
            self.input_width = dimension[3]
        if dimension[3] == 3 or dimension[3] == 1:
            self.channel_first = False
            self.input_channels = dimension[3]
            self.input_height = dimension[1]
            self.input_width = dimension[2]

        self.context = context
        self.engine = engine
        self.stream = stream

    def preprocess_images(self, images):
        assert isinstance(images, list)
        x = []
        for image in images:
            assert image is not None
            image = cv2.resize(
                src=image,
                dsize=(self.input_width, self.input_height),
                interpolation=cv2.INTER_AREA,
            )
            image = np.float32(image)
            image = image * (1 / 255)
            mean = [0.485, 0.456, 0.406]
            # mean = [0.58, 0.55, 0.55]
            std = [0.229, 0.224, 0.225]
            image = (image - mean) / std
            if self.channel_first:
                image = image.transpose((2, 0, 1))
            x.append(image)
        x = np.asarray(x).astype(np.float32)
        return x

    def infer(self, images):
        assert (
            len(images) <= self.engine.max_batch_size
        ), f"[ERROR] Batch size num must be smaller than {self.engine.max_batch_size}"
        threading.Thread.__init__(self)
        # Image preprocessing
        x = self.preprocess_images(images)

        # Create buffers & Allocate images
        bindings = []
        host_inputs = []
        host_outputs = []
        device_inputs = []
        device_outputs = []
        output_shape = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                device_inputs.append(device_mem)
            else:
                output_shape.append(self.engine.get_binding_shape(binding))
                host_outputs.append(host_mem)
                device_outputs.append(device_mem)

        self.cfx.push()
        host_inputs[0] = np.ascontiguousarray(x)

        # Inference
        outputs = []
        cuda.memcpy_htod_async(device_inputs[0], host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=len(images), bindings=bindings, stream_handle=self.stream.handle
        )

        for i in range(len(host_outputs)):
            cuda.memcpy_dtoh_async(host_outputs[i], device_outputs[i], self.stream)
            outputs.append(host_outputs[i].reshape(output_shape[i]))
        self.stream.synchronize()
        self.cfx.pop()

        result = []
        for output in outputs:
            result.append(output[: len(images), :])
        return result

    def __del__(self):
        self.cfx.pop()


class eWaSR_Segmentation:
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.engine = TRTInference(engine_path=self.engine_path)

    def infer_single_image(self, image, resizeTo=None):
        results = self.engine.infer([image])
        out_class = results[1].argmax(1)
        segmentation_output = np.squeeze(out_class).astype("uint8")
        if resizeTo:
            segmentation_output = cv2.resize(
                src=segmentation_output,
                dsize=resizeTo,
                interpolation=cv2.INTER_LINEAR,
            )
        return segmentation_output


if __name__ == "__main__":
    engine_path = "/mnt/onurcan/distance_estimation_deep_learning/utils/pretrained_ewasr/ewasr_resnet18.trt"
    model = eWaSR_Segmentation(engine_path=engine_path)

    for path in sorted(
        Path(
            "/raid/USV/PARSED_5CLASSES/RGB/CVAT_new/folder10/onurcan_stabilized/images"
        ).rglob("*.png")
    ):
        img = cv2.imread(str(path))

        start = time.time()
        segmentation_output = model.infer_single_image(img)
        end = time.time()
        print("{0:.0f}ms".format((end - start) * 1000))

        print("unique vals:", np.unique(segmentation_output))

        cv2.imshow("output_data", segmentation_output * 100)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
