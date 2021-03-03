"""
Copyright (c) 2018-2020 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import namedtuple
import warnings

import torch
import torchvision
import numpy as np

from ..adapters import Adapter
from ..config import BoolField, NumberField, StringField, ConfigValidator, ListField, ConfigError
from ..representation import DetectionPrediction
from ..topology_types import YoloV1Tiny, YoloV2, YoloV2Tiny, YoloV3, YoloV3Tiny, YoloV4, YoloV5
from ..utils import get_or_parse_value

DetectionBox = namedtuple('DetectionBox', ["x", "y", "w", "h", "confidence", "probabilities"])


class YoloOutputProcessor:
    def __init__(self, coord_correct=None, size_correct=None, conf_correct=None,
                 prob_correct=None, coord_normalizer=(1, 1), size_normalizer=(1, 1)):
        self.coord_correct = coord_correct if coord_correct else lambda x: x
        self.size_correct = size_correct if size_correct else np.exp
        self.conf_correct = conf_correct if conf_correct else lambda x: x
        self.prob_correct = prob_correct if prob_correct else lambda x: x
        self.x_normalizer, self.y_normalizer = coord_normalizer
        self.width_normalizer, self.height_normalizer = size_normalizer

    def __call__(self, bbox, i, j, anchors=None):
        if anchors is None:
            anchors = [1, 1]

        # yolov3-onnx, yolov3-tf, yolov4-tf
        x = (self.coord_correct(bbox.x) + i) / self.x_normalizer
        y = (self.coord_correct(bbox.y) + j) / self.y_normalizer
        # onnx
        #x = (2*self.coord_correct(bbox.x)  - 0.5 + i) / self.x_normalizer
        #y = (2*self.coord_correct(bbox.y)  - 0.5 + j) / self.y_normalizer

        # yolov3-onnx, yolov3-tf
        w = self.size_correct(bbox.w) * anchors[0] / self.width_normalizer
        h = self.size_correct(bbox.h) * anchors[1] / self.height_normalizer
        # yolov4-tf, onnx
        #w = (2*self.coord_correct(bbox.w))**2 * anchors[0] / self.width_normalizer
        #h = (2*self.coord_correct(bbox.h))**2 * anchors[1] / self.height_normalizer

        confidence = self.conf_correct(bbox.confidence)
        probabilities = self.prob_correct(bbox.probabilities)

        #print('w = ' + str(w))
        #print('self.size_correct(bbox.w) = ' + str(self.size_correct(bbox.w)) + '   ' + str(bbox.w) + '   ' + str(np.exp(bbox.w)))

        return DetectionBox(x, y, w, h, confidence, probabilities)

class YoloOutputProcessorYoloV4:
    def __init__(self, coord_correct=None, size_correct=None, conf_correct=None,
                 prob_correct=None, coord_normalizer=(1, 1), size_normalizer=(1, 1)):
        self.coord_correct = coord_correct if coord_correct else lambda x: x
        self.size_correct = size_correct if size_correct else np.exp
        self.conf_correct = conf_correct if conf_correct else lambda x: x
        self.prob_correct = prob_correct if prob_correct else lambda x: x
        self.x_normalizer, self.y_normalizer = coord_normalizer
        self.width_normalizer, self.height_normalizer = size_normalizer

    def __call__(self, bbox, i, j, anchors=None):
        if anchors is None:
            anchors = [1, 1]

        # yolov3-onnx, yolov3-tf, yolov4-tf
        x = (self.coord_correct(bbox.x) + i) / self.x_normalizer
        y = (self.coord_correct(bbox.y) + j) / self.y_normalizer
        # onnx
        #x = (2*self.coord_correct(bbox.x)  - 0.5 + i) / self.x_normalizer
        #y = (2*self.coord_correct(bbox.y)  - 0.5 + j) / self.y_normalizer

        # yolov3-onnx, yolov3-tf
        #w = self.size_correct(bbox.w) * anchors[0] / self.width_normalizer
        #h = self.size_correct(bbox.h) * anchors[1] / self.height_normalizer
        # yolov4-tf, onnx
        w = (2*self.coord_correct(bbox.w))**2 * anchors[0] / self.width_normalizer
        h = (2*self.coord_correct(bbox.h))**2 * anchors[1] / self.height_normalizer

        confidence = self.conf_correct(bbox.confidence)
        probabilities = self.prob_correct(bbox.probabilities)

        #print('w = ' + str(w))
        #print('self.size_correct(bbox.w) = ' + str(self.size_correct(bbox.w)) + '   ' + str(bbox.w) + '   ' + str(np.exp(bbox.w)))

        return DetectionBox(x, y, w, h, confidence, probabilities)

class YoloOutputProcessorYoloV5:
    def __init__(self, coord_correct=None, size_correct=None, conf_correct=None,
                 prob_correct=None, coord_normalizer=(1, 1), size_normalizer=(1, 1)):
        self.coord_correct = coord_correct if coord_correct else lambda x: x
        self.size_correct = size_correct if size_correct else np.exp
        self.conf_correct = conf_correct if conf_correct else lambda x: x
        self.prob_correct = prob_correct if prob_correct else lambda x: x
        self.x_normalizer, self.y_normalizer = coord_normalizer
        self.width_normalizer, self.height_normalizer = size_normalizer

    def __call__(self, bbox, i, j, anchors=None):
        if anchors is None:
            anchors = [1, 1]

        # yolov3-onnx, yolov3-tf, yolov4-tf
        #x = (self.coord_correct(bbox.x) + i) / self.x_normalizer
        #y = (self.coord_correct(bbox.y) + j) / self.y_normalizer
        # onnx
        x = (2*self.coord_correct(bbox.x)  - 0.5 + i) / self.x_normalizer
        y = (2*self.coord_correct(bbox.y)  - 0.5 + j) / self.y_normalizer

        # yolov3-onnx, yolov3-tf
        #w = self.size_correct(bbox.w) * anchors[0] / self.width_normalizer
        #h = self.size_correct(bbox.h) * anchors[1] / self.height_normalizer
        # yolov4-tf, onnx
        w = (2*self.coord_correct(bbox.w))**2 * anchors[0] / self.width_normalizer
        h = (2*self.coord_correct(bbox.h))**2 * anchors[1] / self.height_normalizer

        confidence = self.conf_correct(bbox.confidence)
        probabilities = self.prob_correct(bbox.probabilities)

        #print('w = ' + str(w))
        #print('self.size_correct(bbox.w) = ' + str(self.size_correct(bbox.w)) + '   ' + str(bbox.w) + '   ' + str(np.exp(bbox.w)))

        return DetectionBox(x, y, w, h, confidence, probabilities)


class TinyYOLOv1Adapter(Adapter):
    """
    Class for converting output of Tiny YOLO v1 model to DetectionPrediction representation
    """
    __provider__ = 'tiny_yolo_v1'
    prediction_types = (DetectionPrediction, )
    topology_types = (YoloV1Tiny, )

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
             list of DetectionPrediction objects
        """
        prediction = self._extract_predictions(raw, frame_meta)[self.output_blob]

        PROBABILITY_SIZE = 980
        CONFIDENCE_SIZE = 98
        BOXES_SIZE = 392

        CELLS_X, CELLS_Y = 7, 7
        CLASSES = 20
        OBJECTS_PER_CELL = 2

        result = []
        for identifier, output in zip(identifiers, prediction):
            assert PROBABILITY_SIZE + CONFIDENCE_SIZE + BOXES_SIZE == output.shape[0]

            probability, scale, boxes = np.split(output, [PROBABILITY_SIZE, PROBABILITY_SIZE + CONFIDENCE_SIZE])

            probability = np.reshape(probability, (CELLS_Y, CELLS_X, CLASSES))
            scale = np.reshape(scale, (CELLS_Y, CELLS_X, OBJECTS_PER_CELL))
            boxes = np.reshape(boxes, (CELLS_Y, CELLS_X, OBJECTS_PER_CELL, 4))

            confidence = np.zeros((CELLS_Y, CELLS_X, OBJECTS_PER_CELL, CLASSES + 4))
            for cls in range(CLASSES):
                confidence[:, :, 0, cls] = np.multiply(probability[:, :, cls], scale[:, :, 0])
                confidence[:, :, 1, cls] = np.multiply(probability[:, :, cls], scale[:, :, 1])

            labels, scores, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], [], []
            for i, j, k in np.ndindex((CELLS_X, CELLS_Y, OBJECTS_PER_CELL)):
                box = boxes[j, i, k]
                box = [(box[0] + i) / float(CELLS_X), (box[1] + j) / float(CELLS_Y), box[2] ** 2, box[3] ** 2]

                label = np.argmax(confidence[j, i, k, :CLASSES])
                score = confidence[j, i, k, label]

                labels.append(label)
                scores.append(score)
                x_mins.append(box[0] - box[2] / 2.0)
                y_mins.append(box[1] - box[3] / 2.0)
                x_maxs.append(box[0] + box[2] / 2.0)
                y_maxs.append(box[1] + box[3] / 2.0)

            result.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return result


def parse_output(predictions, cells, num, box_size, anchors, processor, threshold=0.001):
    cells_x, cells_y = cells, cells

    labels, scores, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], [], []

    for x, y, n in np.ndindex((cells_x, cells_y, num)):
        if predictions.shape[0] == predictions.shape[1]:
            bbox = predictions[y, x, n*box_size:(n + 1)*box_size]
        else:
            bbox = predictions[n * box_size:(n + 1) * box_size, y, x]

        raw_bbox = DetectionBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5:])
        processed_box = processor(raw_bbox, x, y, anchors[2*n:2*n+2])

        if processed_box.confidence < threshold:
            continue

        classes_prob = processed_box.probabilities
        label = np.argmax(classes_prob)

        labels.append(label)
        scores.append(processed_box.probabilities[label] * processed_box.confidence)
        x_mins.append(processed_box.x - processed_box.w / 2.0)
        y_mins.append(processed_box.y - processed_box.h / 2.0)
        x_maxs.append(processed_box.x + processed_box.w / 2.0)
        y_maxs.append(processed_box.y + processed_box.h / 2.0)

    return labels, scores, x_mins, y_mins, x_maxs, y_maxs

def parse_output_pytorch(prediction, conf_thres=0.001, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [None] * prediction.shape[0]
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

       # best class
        conf, j = x[:, 5:].max(1, keepdim=True)
        
        x = torch.cat((box, conf, j.float()), 1)

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
     
        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        output[xi] = x

    return output

def xywh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

class YoloV2Adapter(Adapter):
    """
    Class for converting output of YOLO v2 family models to DetectionPrediction representation
    """
    __provider__ = 'yolo_v2'
    prediction_types = (DetectionPrediction, )
    topology_types = (YoloV2, YoloV2Tiny, )

    PRECOMPUTED_ANCHORS = {
        'yolo_v2': [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071],
        'tiny_yolo_v2': [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    }

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'classes': NumberField(
                value_type=int, optional=True, min_value=1, default=20, description="Number of detection classes."
            ),
            'coords': NumberField(
                value_type=int, optional=True, min_value=1, default=4, description="Number of bbox coordinates."
            ),
            'num': NumberField(
                value_type=int, optional=True, min_value=1, default=5,
                description="Num parameter from DarkNet configuration file."
            ),
            'anchors': StringField(
                optional=True, choices=YoloV2Adapter.PRECOMPUTED_ANCHORS,
                allow_own_choice=True, default='yolo_v2',
                description="Anchor values provided as comma-separated list or one of precomputed: "
                            "{}".format(', '.join(YoloV2Adapter.PRECOMPUTED_ANCHORS))
            ),
            'cells': NumberField(
                value_type=int, optional=True, min_value=1, default=13,
                description="Number of cells across width and height"
            ),
            'raw_output': BoolField(
                optional=True, default=False,
                description="Indicates, that output is in raw format"
            ),
            'output_format': StringField(
                choices=['BHW', 'HWB'], optional=True, default='BHW',
                description="Set output layer format"
            )
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.classes = self.get_value_from_config('classes')
        self.coords = self.get_value_from_config('coords')
        self.num = self.get_value_from_config('num')
        self.anchors = get_or_parse_value(self.get_value_from_config('anchors'), YoloV2Adapter.PRECOMPUTED_ANCHORS)
        self.cells = self.get_value_from_config('cells')
        self.raw_output = self.get_value_from_config('raw_output')
        self.output_format = self.get_value_from_config('output_format')
        if self.raw_output:
            self.processor = YoloOutputProcessor(coord_correct=lambda x: 1. / (1 + np.exp(-x)),
                                                 conf_correct=lambda x: 1. / (1 + np.exp(-x)),
                                                 prob_correct=lambda x: np.exp(x) / np.sum(np.exp(x)),
                                                 coord_normalizer=(self.cells, self.cells),
                                                 size_normalizer=(self.cells, self.cells))
        else:
            self.processor = YoloOutputProcessor(coord_normalizer=(self.cells, self.cells),
                                                 size_normalizer=(self.cells, self.cells))

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """
        predictions = self._extract_predictions(raw, frame_meta)[self.output_blob]

        result = []
        box_size = self.classes + self.coords + 1
        for identifier, prediction in zip(identifiers, predictions):
            if len(prediction.shape) != 3:
                if self.output_format == 'BHW':
                    new_shape = (self.num * box_size, self.cells, self.cells)
                else:
                    new_shape = (self.cells, self.cells, self.num * box_size)
                prediction = np.reshape(prediction, new_shape)
            labels, scores, x_mins, y_mins, x_maxs, y_maxs = parse_output(prediction, self.cells, self.num,
                                                                          box_size, self.anchors,
                                                                          self.processor)

            result.append(DetectionPrediction(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs))

        return result


class YoloV3Adapter(Adapter):
    """
    Class for converting output of YOLO v3 family models to DetectionPrediction representation
    """
    __provider__ = 'yolo_v3'
    prediction_types = (DetectionPrediction, )
    topology_types = (YoloV3, YoloV3Tiny, )

    PRECOMPUTED_ANCHORS = {
        'yolo_v3': [
            10.0, 13.0,
            16.0, 30.0,
            33.0, 23.0,
            30.0, 61.0,
            62.0, 45.0,
            59.0, 119.0,
            116.0, 90.0,
            156.0, 198.0,
            373.0, 326.0
        ],
        'tiny_yolo_v3': [
            10.0, 14.0,
            23.0, 27.0,
            37.0, 58.0,
            81.0, 82.0,
            135.0, 169.0,
            344.0, 319.0
        ]
    }

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'classes': NumberField(
                value_type=int, optional=True, min_value=1, default=80, description="Number of detection classes."
            ),
            'coords': NumberField(
                value_type=int, optional=True, min_value=1, default=4, description="Number of bbox coordinates."
            ),
            'num': NumberField(
                value_type=int, optional=True, min_value=1, default=3,
                description="Num parameter from DarkNet configuration file."
            ),
            'anchors': StringField(
                optional=True, choices=YoloV3Adapter.PRECOMPUTED_ANCHORS.keys(), allow_own_choice=True,
                default='yolo_v3',
                description="Anchor values provided as comma-separated list or one of precomputed: "
                            "{}.".format(', '.join(YoloV3Adapter.PRECOMPUTED_ANCHORS.keys()))),
            'threshold': NumberField(value_type=float, optional=True, min_value=0, default=0.001,
                                     description="Minimal objectiveness score value for valid detections."),
            'outputs': ListField(description="The list of output layers names."),
            'anchor_masks': ListField(optional=True, description='per layer used anchors mask'),
            'do_reshape': BoolField(
                optional=True, default=False,
                description="Reshapes output tensor to [B,Cy,Cx] or [Cy,Cx,B] format, depending on 'output_format'"
                            "value ([B,Cy,Cx] by default). You may need to specify 'cells' value."
            ),
            'cells': ListField(
                optional=True, default=[13, 26, 52],
                description="Grid size for each layer, according 'outputs' filed. Works only with 'do_reshape=True' or "
                            "when output tensor dimensions not equal 3."),
            'raw_output': BoolField(
                optional=True, default=False,
                description="Preprocesses output in the original way."
            ),
            'output_format': StringField(
                choices=['BHW', 'HWB'], optional=True, default='BHW',
                description="Set output layer format"
            )
        })

        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.classes = self.get_value_from_config('classes')
        self.coords = self.get_value_from_config('coords')
        self.num = self.get_value_from_config('num')
        self.anchors = get_or_parse_value(self.get_value_from_config('anchors'), YoloV3Adapter.PRECOMPUTED_ANCHORS)
        self.threshold = self.get_value_from_config('threshold')
        self.outputs = self.get_value_from_config('outputs')
        anchor_masks = self.get_value_from_config('anchor_masks')
        self.masked_anchors = None
        if anchor_masks is not None:
            per_layer_anchors = []
            for layer_mask in anchor_masks:
                layer_anchors = []
                for idx in layer_mask:
                    layer_anchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
                per_layer_anchors.append(layer_anchors)
            self.masked_anchors = per_layer_anchors
        self.do_reshape = self.get_value_from_config('do_reshape')
        self.cells = self.get_value_from_config('cells')
        if len(self.outputs) != len(self.cells):
            if self.do_reshape:
                raise ConfigError('Incorrect number of output layer ({}) or detection grid size ({}). '
                                  'Must be equal with each other, check "cells" or "outputs" option'
                                  .format(len(self.outputs), len(self.cells)))
            warnings.warn('Number of output layers ({}) not equal to detection grid size ({}). '
                          'Must be equal with each other, if output tensor resize is required'
                          .format(len(self.outputs), len(self.cells)))

        if self.masked_anchors and len(self.masked_anchors) != len(self.outputs):
            raise ConfigError('anchor mask should be specified for all output layers')

        self.raw_output = self.get_value_from_config('raw_output')
        self.output_format = self.get_value_from_config('output_format')
        if self.raw_output:
            self.processor = YoloOutputProcessor(coord_correct=lambda x: 1.0 / (1.0 + np.exp(-x)),
                                                 conf_correct=lambda x: 1.0 / (1.0 + np.exp(-x)),
                                                 prob_correct=lambda x: 1.0 / (1.0 + np.exp(-x)))
        else:
            self.processor = YoloOutputProcessor()

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """

        result = []

        raw_outputs = self._extract_predictions(raw, frame_meta)
        batch = len(identifiers)
        predictions = [[] for _ in range(batch)]
        for blob in self.outputs:
            for b in range(batch):
                predictions[b].append(raw_outputs[blob][b])

        box_size = self.coords + 1 + self.classes
        for identifier, prediction, meta in zip(identifiers, predictions, frame_meta):
            detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
            input_shape = list(meta.get('input_shape', {'data': (1, 3, 416, 416)}).values())[0]
            nchw_layout = input_shape[1] == 3
            self.processor.width_normalizer = input_shape[3 if nchw_layout else 2]
            self.processor.height_normalizer = input_shape[2 if nchw_layout else 1]
            for layer_id, p in enumerate(prediction):
                anchors = self.masked_anchors[layer_id] if self.masked_anchors else self.anchors
                num = len(anchors) // 2 if self.masked_anchors else self.num
                if self.do_reshape or len(p.shape) != 3:
                    try:
                        cells = self.cells[layer_id]
                    except IndexError:
                        raise ConfigError('Number of output layers ({}) is more than detection grid size ({}). '
                                          'Check "cells" option.'.format(len(outputs), len(self.cells)))
                    if self.output_format == 'BHW':
                        new_shape = (num * box_size, cells, cells)
                    else:
                        new_shape = (cells, cells, num * box_size)
                    p = np.reshape(p, new_shape)
                else:
                    # Get grid size from output shape - ignore self.cells value.
                    # N.B.: value p.shape[1] will always contain grid size, but here we use if clause just for
                    # clarification (works ONLY for square grids).
                    cells = p.shape[1] if self.output_format == 'BHW' else p.shape[0]

                self.processor.x_normalizer = cells
                self.processor.y_normalizer = cells
                
                labels, scores, x_mins, y_mins, x_maxs, y_maxs = parse_output(p, cells, num,
                                                                              box_size, anchors,
                                                                              self.processor, self.threshold)
                detections['labels'].extend(labels)
                detections['scores'].extend(scores)
                detections['x_mins'].extend(x_mins)
                detections['y_mins'].extend(y_mins)
                detections['x_maxs'].extend(x_maxs)
                detections['y_maxs'].extend(y_maxs)

            result.append(DetectionPrediction(
                identifier, detections['labels'], detections['scores'], detections['x_mins'], detections['y_mins'],
                detections['x_maxs'], detections['y_maxs']
            ))

        return result

class YoloV3ONNX(Adapter):
    __provider__ = 'yolo_v3_onnx'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'boxes_out': StringField(),
            'scores_out': StringField(),
            'indices_out': StringField()
        })
        return params

    def configure(self):
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.scores_out = self.get_value_from_config('scores_out')
        self.indices_out = self.get_value_from_config('indices_out')

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        result = []
        indicies_out = raw_outputs[self.indices_out]
        if len(indicies_out.shape) == 2:
            indicies_out = np.expand_dims(indicies_out, 0)
        for identifier, boxes, scores, indices in zip(
                identifiers, raw_outputs[self.boxes_out], raw_outputs[self.scores_out], indicies_out
        ):
            out_boxes, out_scores, out_classes = [], [], []
            for idx_ in indices:
                if idx_[0] == -1:
                    break
                out_classes.append(idx_[1])
                out_scores.append(scores[tuple(idx_[1:])])
                out_boxes.append(boxes[idx_[2]])
            transposed_boxes = np.array(out_boxes).T if out_boxes else ([], [], [], [])
            x_mins = transposed_boxes[1]
            y_mins = transposed_boxes[0]
            x_maxs = transposed_boxes[3]
            y_maxs = transposed_boxes[2]
            result.append(DetectionPrediction(identifier, out_classes, out_scores, x_mins, y_mins, x_maxs, y_maxs))
        return result


class PytorchYoloAdapter(Adapter):
    """
    Class for converting output of YOLO v3 family models to DetectionPrediction representation
    """
    __provider__ = 'pytorch_yolo'
    prediction_types = (DetectionPrediction, )
    topology_types = (YoloV4, YoloV5)

    PRECOMPUTED_ANCHORS = {
        'yolo_v4': [
            12.0, 16.0,
            19.0, 36.0,
            40.0, 28.0,
            36.0, 75.0,
            76.0, 55.0,
            72.0, 146.0,
            142.0, 110.0,
            192.0, 243.0,
            459.0, 401.0
        ],
        'yolo_v5': [
            10.0, 13.0,
            16.0, 30.0,
            33.0, 23.0,
            30.0, 61.0,
            62.0, 45.0,
            59.0, 119.0,
            116.0, 90.0,
            156.0, 198.0,
            373.0, 326.0
        ]
    }

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'threshold': NumberField(value_type=float, optional=True, min_value=0, default=0.001,
                                     description="Minimal objectiveness score value for valid detections."),
            'outputs': ListField(description="The list of output layers names."),
            'anchors': StringField(
                optional=True, choices=YoloV3Adapter.PRECOMPUTED_ANCHORS.keys(), allow_own_choice=True,
                default='yolo_v4',
                description="Anchor values provided as comma-separated list or one of precomputed: "
                            "{}.".format(', '.join(YoloV3Adapter.PRECOMPUTED_ANCHORS.keys()))),
            'num': NumberField(
                value_type=int, optional=True, min_value=1, default=3,
                description="Num parameter from DarkNet configuration file."
            ),
            'anchor_masks': ListField(optional=True, description='per layer used anchors mask'),
            'raw_output': BoolField(
                optional=True, default=False,
                description="Preprocesses output in the original way."
            )
        })

        return parameters

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.num = self.get_value_from_config('num')
        self.threshold = self.get_value_from_config('threshold')
        self.outputs = self.get_value_from_config('outputs')
        self.anchors = get_or_parse_value(self.get_value_from_config('anchors'), PytorchYoloAdapter.PRECOMPUTED_ANCHORS)
        self.raw_output = self.get_value_from_config('raw_output')
        anchor_masks = self.get_value_from_config('anchor_masks')
        self.masked_anchors = None
        if anchor_masks is not None:
            per_layer_anchors = []
            for layer_mask in anchor_masks:
                layer_anchors = []
                for idx in layer_mask:
                    layer_anchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
                per_layer_anchors.append(layer_anchors)
            self.masked_anchors = per_layer_anchors

        if self.masked_anchors and len(self.masked_anchors) != len(self.outputs):
            raise ConfigError('anchor mask should be specified for all output layers')

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
        Returns:
            list of DetectionPrediction objects
        """

        result = []

        raw_outputs = self._extract_predictions(raw, frame_meta)

        batch = len(identifiers)

        _, _, size, _ = list(frame_meta[0].get('input_shape', {'data': (1, 3, 416, 416)}).values())[0]

        predictions = []

        for blob in self.outputs:

            prediction = raw_outputs[blob]
            prediction = 1. / (1 + np.exp(-1*prediction))
            bs, _, ny, nx = prediction.shape

            if int(size/nx) == 8 & int(size/ny) == 8: #80x80, 
                idx = 0
            elif int(size/nx) == 16 & int(size/ny) == 16: #40x40
                idx = 1
            elif int(size/nx) == 32 & int(size/ny) == 32: # 20x20
                idx = 2
            elif int(size/nx) == 64 & int(size/ny) == 64: #
                idx = 3
            elif int(size/nx) == 128 & int(size/ny) == 128: #
                idx = 4

            anchors = self.masked_anchors[idx] if self.masked_anchors else self.anchors
            num = len(anchors) // 2 if self.masked_anchors else self.num

            for n in range(num):
                for row, col in np.ndindex((ny, nx)):
                    #bbox = r[0, i, 85, row, col]
                    bbox = prediction[0, n*85:(n+1)*85, row, col]
                    x, y, width, height, object_probability = bbox[:5]
                    class_probabilities = bbox[5:]
                    
                    # yolo-v4-tf
                    #####################################
                    #bbox[0] = (x + col) *(size / nx)
                    #bbox[1] = (y + row) *(size / ny)
                    #####################################
                    # pytorch yolo
                    bbox[0] = (2*x - 0.5 + col)*(size / nx)
                    bbox[1] = (2*y - 0.5 + row)*(size / ny)
                    #####################################

                    bbox[2] = (2*width)**2* anchors[2 * n] 
                    bbox[3] = (2*height)**2 * anchors[2 * n + 1]
            
                    prediction[0, n*85:(n+1)*85, row, col] = bbox
            predictions.append(prediction)

        if len(self.outputs) == 3:
            pred_bbox = np.concatenate([predictions[0].reshape(1, num, -1, predictions[0].shape[2]* predictions[0].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85),
                                        predictions[1].reshape(1, num, -1, predictions[1].shape[2]* predictions[1].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85),
                                        predictions[2].reshape(1, num, -1, predictions[2].shape[2]* predictions[2].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85)], axis=1)
        elif len(self.outputs) == 4:
            pred_bbox = np.concatenate([predictions[0].reshape(1, num, -1, predictions[0].shape[2]* predictions[0].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85),
                                        predictions[1].reshape(1, num, -1, predictions[1].shape[2]* predictions[1].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85),
                                        predictions[2].reshape(1, num, -1, predictions[2].shape[2]* predictions[2].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85),
                                        predictions[3].reshape(1, num, -1, predictions[3].shape[2]* predictions[3].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85)], axis=1)

        elif len(self.outputs) == 5:
            pred_bbox = np.concatenate([predictions[0].reshape(1, num, -1, predictions[0].shape[2]* predictions[0].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85),
                                        predictions[1].reshape(1, num, -1, predictions[1].shape[2]* predictions[1].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85),
                                        predictions[2].reshape(1, num, -1, predictions[2].shape[2]* predictions[2].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85),
                                        predictions[3].reshape(1, num, -1, predictions[3].shape[2]* predictions[3].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85),
                                        predictions[4].reshape(1, num, -1, predictions[4].shape[2]* predictions[4].shape[3]).transpose(0, 1, 3, 2).reshape(1, -1, 85)], axis=1)


        img_tensor= torch.from_numpy(pred_bbox)

        # Apply NMS
        pred = parse_output_pytorch(img_tensor, self.threshold, 0, merge=False, classes=None, agnostic=False)
        
        detections = {'labels': [], 'scores': [], 'x_mins': [], 'y_mins': [], 'x_maxs': [], 'y_maxs': []}
        
        

        if  pred[0] is not None:
            data=pred[0].numpy()    
            for i, proposal in enumerate(data):
                
                confidence = proposal[4]
                xmin = np.float( (proposal[0]/size))
                ymin = np.float( (proposal[1]/size))
                xmax = np.float( (proposal[2]/size))
                ymax = np.float( (proposal[3]/size))
                idx = np.int(proposal[5])
     
                detections['labels'].append(idx)
                detections['scores'].append(confidence)
                detections['x_mins'].append(xmin)
                detections['y_mins'].append(ymin)
                detections['x_maxs'].append(xmax)
                detections['y_maxs'].append(ymax)
 
        result.append(DetectionPrediction(
            identifiers[0], detections['labels'], detections['scores'], detections['x_mins'], detections['y_mins'],
            detections['x_maxs'], detections['y_maxs']
            ))

        return result