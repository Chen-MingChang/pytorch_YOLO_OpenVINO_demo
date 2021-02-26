# pytorch_YOLO_OpenVINO_demo

Integration of Pytorch YOLO models (YOLO-V3 / YOLO-V4 / Scaled-YOLO-V4 / YOLO-V5) for OpenVINO inference.
Windows 10 and Ubuntu 18.04 are validated to use. (Scaled-YOLO-V4 is only available on Ubuntu)

## Convert Weights to ONNX File
The following components are required.
-	OpenVINO 2021.2
-	ONNX >= 1.8.0
-	Pytorch >= 1.7.0
-	Netron 4.4.3
-	See 'requirements.txt' for other required components

###	Download Pytorch Weights 

About YOLOV3
This document uses a darknet YOLOV3 model, because the author has not found a public pytorch version.
https://github.com/zldrobit/onnx_tflite_yolov3

About YOLOV4
Taken from the repository released by one of the authors of YOLOV4, KinYiu Wong.
https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u3_preview
There are two kind of models in the repository; all the activation of  convolutional layer of yolov4-pacsp uses leaky, and yolov4-pacsp-mish uses mish at activation.

About Scaled-YOLOV4
Taken from the repository released by one of the authors of Scaled-YOLOV4, KinYiu Wong.
https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-large#installation
There are three models (p5, p6, p7)with different backbone.

About YOLOV5
There are Four tag in YOLOv5 repository so far. And YOLOv5 includes YOLOv5s, YOLOv5m, YOLOv5l and YOLOv5x due to different backbone. Here we use models from tag v4.0 for inference. 
https://github.com/ultralytics/yolov5

| Model | Test Size | weights |
| :-- | :-: | :-: | 
|  |  |  |
| **YOLOv3** | 416 | [weights](https://pjreddie.com/media/files/yolov3.weights) |
|  |  |  |
| **YOLOv4**<sub>pacsp</sub> | 672 | [weights](https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/releases/download/models/yolov4-pacsp.pt) |
| **YOLOv4**<sub>pacsp-mish</sub> | 672 | [weights](https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/releases/download/models/yolov4-pacsp-mish.pt) |
|  |  |  |
| **Scaled-YOLOv4-p5** | 896 | [weights](https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/releases/download/models/yolov4-p5.pt) |
| **Scaled-YOLOv4-p5** | 1280 | [weights](https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/releases/download/models/yolov4-p6.pt) |
| **Scaled-YOLOv4-p5** | 1536 | [weights](https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/releases/download/models/yolov4-p7.pt) |
|  |  |  |
| **YOLOv5s** | 640 | [weights](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt) |
| **YOLOv5m** | 640 | [weights](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5m.pt) |
| **YOLOv5l** | 640 | [weights](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5l.pt) |
| **YOLOv5x** | 640 | [weights](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5x.pt) |

