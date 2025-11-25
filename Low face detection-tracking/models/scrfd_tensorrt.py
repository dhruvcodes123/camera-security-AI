import os
import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional

from .tensorrt_model import TensorRTModel
from src.utils.helpers import distance2bbox, distance2kps

__all__ = ["SCRFD_TensorRT"]


class SCRFD_TensorRT:
    """
    TensorRT-optimized SCRFD Face Detector for Jetson
    
    Title: "Sample and Computation Redistribution for Efficient Face Detection"
    Paper: https://arxiv.org/abs/2105.04714
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int] = (640, 640),
        conf_thres: float = 0.5,
        iou_thres: float = 0.4
    ) -> None:
        """SCRFD TensorRT initialization

        Args:
            model_path (str): Path to ONNX model file (will be converted to TensorRT if needed)
            input_size (int): Input image size. Defaults to (640, 640)
            conf_thres (float, optional): Confidence threshold. Defaults to 0.5.
            iou_thres (float, optional): Non-max supression (NMS) threshold. Defaults to 0.4.
        """
        self.logger = logging.getLogger("SCRFD_TensorRT")
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # SCRFD model params --------------
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

        self.mean = 127.5
        self.std = 128.0

        self.center_cache = {}
        # ---------------------------------

        # Check for TensorRT engine file
        engine_path = model_path.replace('.onnx', '.engine')
        
        # Initialize TensorRT model
        self._initialize_model(onnx_path=model_path, engine_path=engine_path)

    def _initialize_model(self, onnx_path: str, engine_path: str):
        """Initialize the TensorRT model from the given path.

        Args:
            onnx_path (str): Path to .onnx model.
            engine_path (str): Path to .engine file (will be created if it doesn't exist).
        """
        try:
            self.model = TensorRTModel(engine_path=engine_path, onnx_path=onnx_path)
            self.logger.info("TensorRT SCRFD model initialized successfully")
            
            # Set output names based on ONNX model structure
            # For SCRFD, outputs are organized as:
            # - First fmc outputs: classification scores
            # - Next fmc outputs: bounding box predictions
            # - Last fmc outputs (if use_kps): keypoint predictions
            self.output_names = []
            for i in range(self.fmc):
                self.output_names.append(f"score_8_{i}")
            for i in range(self.fmc):
                self.output_names.append(f"bbox_8_{i}")
            if self.use_kps:
                for i in range(self.fmc):
                    self.output_names.append(f"kps_8_{i}")
                    
            self.input_name = "input"
            
        except Exception as e:
            self.logger.error(f"Failed to load the TensorRT model: {e}")
            raise

    def forward(self, image, threshold):
        """
        Run inference on the image
        
        Args:
            image: Input image
            threshold: Detection confidence threshold
            
        Returns:
            scores_list, bboxes_list, kpss_list
        """
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(image.shape[0:2][::-1])

        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / self.std,
            input_size,
            (self.mean, self.mean, self.mean),
            swapRB=True
        )
        
        # Run inference
        outputs = self.model.infer(blob)
        
        # Process outputs
        input_height = blob.shape[2]
        input_width = blob.shape[3]

        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = outputs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = self._get_anchor_centers(height, width, stride)
                self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            if len(pos_inds) == 0:
                continue

            scores = scores[pos_inds]
            bbox_preds = bbox_preds[pos_inds, :]
            anchor_centers = anchor_centers[pos_inds, :]

            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores)
            bboxes_list.append(bboxes)

            if self.use_kps:
                kps_preds = kps_preds[pos_inds, :]
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss_list.append(kpss)

        return scores_list, bboxes_list, kpss_list

    def _get_anchor_centers(self, height, width, stride):
        """Generate anchor centers for feature map."""
        y, x = np.mgrid[:height, :width]
        y = (y * stride).reshape(-1, 1)
        x = (x * stride).reshape(-1, 1)
        anchor_centers = np.hstack((x, y))
        return anchor_centers

    def detect(self, img, max_num=0, metric='default'):
        """
        Detect faces in the image
        
        Args:
            img: Input image
            max_num: Maximum number of detections (0 for unlimited)
            metric: Sorting metric ('default' or 'max_size')
            
        Returns:
            bboxes, kpss (if use_kps is True)
        """
        input_size = self.input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.conf_thres)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self._nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max_size':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def _nms(self, dets):
        """Non-Maximum Suppression"""
        thresh = self.iou_thres
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep 