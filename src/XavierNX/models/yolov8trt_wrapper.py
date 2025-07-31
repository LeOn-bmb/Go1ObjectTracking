import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class YOLOv8TensorRT:
    def __init__(self, engine_path, input_width, input_height, conf_thresh, iou_thresh):
        self.engine_path = engine_path
        self.input_width = input_width
        self.input_height = input_height
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_idx = self.engine.get_binding_index("images") if "images" in self.engine else 0
        self.output_idx = 1 - self.input_idx
        self.input_shape = (1, 3, input_height, input_width)

        self.context.set_binding_shape(self.input_idx, self.input_shape)

        self.input_size = trt.volume(self.input_shape) * np.float32().itemsize
        self.d_input = cuda.mem_alloc(self.input_size)
        self.host_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)

        self.output_shape = tuple(self.context.get_binding_shape(self.output_idx))  # dynamic
        self.output_size = trt.volume(self.output_shape) * np.float32().itemsize
        self.d_output = cuda.mem_alloc(self.output_size)
        self.host_output = cuda.pagelocked_empty(trt.volume(self.output_shape), dtype=np.float32)

        self.bindings = [int(self.d_input), int(self.d_output)]

    def preprocess(self, image):
        # Originalbild-Größe
        h0, w0 = image.shape[:2]
        r = min(self.input_width / w0, self.input_height / h0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))

        # Resize mit intakter Aspect Ratio
        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Padding hinzufügen
        dw = self.input_width - new_unpad[0]
        dh = self.input_height - new_unpad[1]
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Debug: merken für spätere Rücktransformation
        self.letterbox_info = (r, left, top)

        img = padded.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img.ravel()

    def infer(self, image):
        self.host_input[:] = self.preprocess(image)
        cuda.memcpy_htod(self.d_input, self.host_input)
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(self.host_output, self.d_output)
        return self.postprocess(self.host_output, image.shape)

    def postprocess(self, output, original_shape):
        output = np.asarray(self.host_output).reshape(6, -1).T  # (4095, 6)

        # Jetzt enthält jede Zeile: x, y, w, h, conf, class_id
        boxes = output[:, :4]
        confidences = output[:, 4]
        class_ids = output[:, 5].astype(int)

        # Optional: Score-Filter
        mask = confidences > self.conf_thresh
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # xywh → xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # Rücktransformation
        r, left, top = self.letterbox_info
        boxes_xyxy[:, [0, 2]] -= left
        boxes_xyxy[:, [1, 3]] -= top
        boxes_xyxy /= r

        # Skalierung auf Originalbildgröße
        h_orig, w_orig = original_shape[:2]
        boxes_xyxy[:, [0, 2]] *= w_orig / self.input_width
        boxes_xyxy[:, [1, 3]] *= h_orig / self.input_height

        # Clipping auf Bildränder
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w_orig)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h_orig)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w_orig)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h_orig)

        result = []
        for i in range(len(boxes_xyxy)):
            box = boxes_xyxy[i]
            result.append([box[0], box[1], box[2], box[3], confidences[i], class_ids[i]])
        return result

    def nms(self, boxes, scores, iou_threshold):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return keep