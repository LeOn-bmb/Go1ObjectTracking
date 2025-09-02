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
        # BGR → RGB (Training auf RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]

        # Verhältnis berechnen, damit Aspect Ratio erhalten bleibt
        r = min(self.input_width / w0, self.input_height / h0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Padding berechnen
        dw = self.input_width - new_w
        dh = self.input_height - new_h
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2

        # Letterbox mit grauem Hintergrund
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Normalisierung auf [0,1]
        img = padded.astype(np.float32) / 255.0

        # Channel-First
        img = img.transpose(2, 0, 1)

        # Batch-Dimension hinzufügen
        img = np.expand_dims(img, axis=0)
        return img.ravel(), (r, left, top)


    def infer(self, image):
        img, ratio_pad = self.preprocess(image)   # entpacken
        self.host_input[:] = img.ravel()          # nur das Bild kopieren
        cuda.memcpy_htod(self.d_input, self.host_input)
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(self.host_output, self.d_output)
        return self.postprocess(self.host_output, image.shape, ratio_pad)

    def postprocess(self, output, original_shape, ratio_pad):
        r, pad_x, pad_y = ratio_pad
        output = np.asarray(self.host_output).reshape(1, 6, -1)
        output = np.squeeze(output, axis=0).T   # (4095, 6)

        boxes = output[:, :4]
        confidences = output[:, 4]
        class_ids = output[:, 5].astype(int)

        # Score-Filter
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

        # Rückskalierung auf Originalbild
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / r
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / r

        # Clipping
        h_orig, w_orig = original_shape[:2]
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w_orig)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h_orig)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w_orig)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h_orig)

        # NMS
        if len(boxes_xyxy) > 0:
            keep = self.nms(boxes_xyxy, confidences, self.iou_thresh)
            boxes_xyxy = boxes_xyxy[keep]
            confidences = confidences[keep]
            class_ids = class_ids[keep]

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