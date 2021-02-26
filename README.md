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
There are three tag in YOLOv5 repository so far. And YOLOv5 includes YOLOv5s, YOLOv5m, YOLOv5l and YOLOv5x due to different backbone. Here we use YOLOv5s from tag v3.0 for description. Run the following command to download yolov5s.pt:

```
$ wget https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5s.pt 
```

| Model | Test Size | weights |
| :-- | :-: | :-: | 
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
|  |  |  |
