# pytorch_YOLO_OpenVINO_demo

Integration of Pytorch YOLO models (YOLO-V3 / YOLO-V4 / Scaled-YOLO-V4 / YOLO-V5) for OpenVINO inference.  
Windows 10 and Ubuntu 18.04 are validated to use. (Scaled-YOLO-V4 is only available on Ubuntu)

## Convert Weights to ONNX File
The following components are required.
-	OpenVINO 2021.2
-	Python >=  3.6
-	ONNX >= 1.8.0
-	Pytorch >= 1.7.0
-	Netron 4.4.3
-	See 'requirements.txt' for other required components

###	Download Pytorch Weights 

**About YOLOV3**  
This document uses a darknet YOLOV3 model, because the author has not found a public pytorch version.  
https://github.com/zldrobit/onnx_tflite_yolov3

**About YOLOV4**  
Taken from the repository released by one of the authors of YOLOV4, KinYiu Wong.  
https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u3_preview  
Take two models as examples from the repository; all the activation of  convolutional layer of yolov4-pacsp uses leaky, and yolov4-pacsp-mish uses mish at activation.

**About Scaled-YOLOV4**  
Taken from the repository released by one of the authors of Scaled-YOLOV4, KinYiu Wong.  
https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-large#installation  
There are three scaled models (p5, p6, p7) with different backbone.

**About YOLOV5**  
There are four tag in YOLOv5 repository so far. And YOLOv5 includes YOLOv5s, YOLOv5m, YOLOv5l and YOLOv5x due to different backbone. Here we use models from tag v4.0 for inference.  
https://github.com/ultralytics/yolov5

| Model | Test Size | Weights | AP<sub>50</sub><sup>OpenVINO Accuracy Checker Tool</sup>
| :-- | :-: | :-: | :-: | 
|  |  |  |  |
| **YOLOv3** | 416 | [weights](https://pjreddie.com/media/files/yolov3.weights) | 60.23% |
|  |  |  |  |
| **YOLOv4**<sub>pacsp</sub> | 672 | [weights](https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/releases/download/models/yolov4-pacsp.pt) | 65.71% |
| **YOLOv4**<sub>pacsp-mish</sub> | 672 | [weights](https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/releases/download/models/yolov4-pacsp-mish.pt) | 65.41% |
|  |  |  |  |
| **Scaled-YOLOv4-p5** | 896 | [weights](https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/releases/download/models/yolov4-p5.pt) | 69.06% |
| **Scaled-YOLOv4-p6** | 1280 | [weights](https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/releases/download/models/yolov4-p6.pt) | 71.39% |
| **Scaled-YOLOv4-p7** | 1536 | [weights](https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/releases/download/models/yolov4-p7.pt) | 73.11% |
|  |  |  |  |
| **YOLOv5s** | 640 | [weights](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s.pt) | 50.85% |
| **YOLOv5m** | 640 | [weights](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5m.pt) | 59.45% |
| **YOLOv5l** | 640 | [weights](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5l.pt) | 62.85% |
| **YOLOv5x** | 640 | [weights](https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5x.pt) | 64.79% |

###	Convert Pytorch Weights to ONNX Weights 

#### If YOLOV3: You can following https://github.com/zldrobit/onnx_tflite_yolov3 to convert model to onnx.  
```
  Edited detect.py (optional) to obtain the weight with higher accuracy.  
  line 48 'opset_version=9' -> 'opset_version=12'
  
$ python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights
```

#### If YOLOV4 / Scaled-YOLOV4 / YOLOV5:  
The repository provides a script **models/export.py** to export Pytorch weights with extensions *.pt to ONNX weights with extensions *.onnx.  

Intall mish-cuda if you want to convert Scaled-YOLOV4. (refer to https://github.com/DataXujing/ScaledYOLOv4)  

```
$ git clone https://github.com/thomasbrandon/mish-cuda mc
$ cd mc

  change all of name which is mish_cuda to mish_mish and build.
  1. mc/src/mish_cuda -> mc/src/mish_mish
  2. mc/csrc/mish_cuda.cpp -> mc/csrc/mish_mish.cpp
  3. in mc/setup.py
    3.1 line 5 -> 'csrc/mish_mish.cpp'
    3.2 line 11 -> name='mish_mish'
    3.3 line 20 -> 'mish_mish._C'

$ python3 setup.py build
  rename mc/build/lib.xxx folder to mc/build/lib
```

Next, edit **models/yolo.py** and **models/common.py** according to the model to be converted.  
There is one part that needs to be edited at line.46 in **models/yolo.py**,  

```
# comment out only if yolov4, (scaled-yolov4 and yolov5 do not need)
            x[i] = self.m[i](x[i])  # conv
```

and there are two at line.42-line.45 and line.80-line.83 in **models/common.py**.  

```
# if yolov4
        #self.act = Mish() if act else nn.Identity()
# if yolov5
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
```

```
# if yolov4
        #self.act = Mish()
# if yolov5
        self.act = nn.LeakyReLU(0.1, inplace=True)
```

At last, run **models/export.py** to generate *.onnx* file.  
Here take yolov5s.pt (default) as an example, run the following command:

```
$ python3 models/export.py  --weights yolov5s.pt  --img-size 640
```

Then we can get yolov5s.onnx.


##	Convert ONNX File to IR File
After we get ONNX weights file from the last section, we can convert it to IR file with model optimizer.  
(refer to https://github.com/violet17/yolov5_demo)  
Run the following script to temporarily set OpenVINO environment and variables:

```
$ source /opt/intel/openvino_2021/bin/setupvars.sh
```

We need to specify the output node of the IR when we use model optimizer to convert the ONNX weights file.  
For example, there are 3 output nodes in yolov5s.onnx that obtained in the previous step. We can use Netron to visualize yolov5s.onnx. Then we find the output nodes by searching the keyword “Transpose” in Netron. After that, we can find the convolution node marked as oval shown in following Figure. After double clicking the convolution node, we can read its name “Conv_245”.  

<img src="https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/blob/main/yolov5s_output_node.png" width="70%">

Similarly, we can find the other two output nodes “Conv_261” and “Conv_277”.  

we can run the following command to generate the IR of YOLOv5 model:
```
$ python3 /opt/intel/openvino_2021.1.110/deployment_tools/model_optimizer/mo.py  --input_model yolov5s.onnx -s 255 --reverse_input_channels --output Conv_245,Conv_261,Conv_277
```

After that, we can get IR of yolov5s in FP32.  

##	OpenVINO Inference Python Demo

After generate IR model, we can use **yolo__openvino_demo.py** for inference.  
The usage is as follows:  

```
usage: yolo__openvino_demo.py [-h] -m MODEL -at
                              {yolov3,yolov4,yolov5,yolov4-p5,yolov4-p6,yolov4-p7}
                              -i INPUT [-l CPU_EXTENSION] [-d DEVICE]
                              [--labels LABELS] [-t PROB_THRESHOLD]
                              [-iout IOU_THRESHOLD] [-ni NUMBER_ITER] [-pc]
                              [-r] [--no_show]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -at {yolov3,yolov4,yolov5,yolov4-p5,yolov4-p6,yolov4-p7}, --architecture_type {yolov3,yolov4,yolov5,yolov4-p5,yolov4-p6,yolov4-p7}
                        Required. Specify model' architecture type.
  -i INPUT, --input INPUT
                        Required. Path to an image/video file. (Specify 'cam'
                        to work with camera)
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. Absolute
                        path to a shared library with the kernels
                        implementations.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL or MYRIAD is acceptable. The sample
                        will look for a suitable plugin for device specified.
                        Default value is CPU
  --labels LABELS       Optional. Labels mapping file
  -t PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Optional. Probability threshold for detections
                        filtering
  -iout IOU_THRESHOLD, --iou_threshold IOU_THRESHOLD
                        Optional. Intersection over union threshold for
                        overlapping detections filtering
  -ni NUMBER_ITER, --number_iter NUMBER_ITER
                        Optional. Number of inference iterations
  -pc, --perf_counts    Optional. Report performance counters
  -r, --raw_output_message
                        Optional. Output inference results raw values showing
  --no_show             Optional. Don't show output
```
  
Here we take yolov5s as an example, we can use the following command to run the demo.
  
```
$ python3 yolo__openvino_demo.py -m yolov5s.xml -i images/bus.jpg -at yolov5
```
<img src="https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/blob/main/demo_result.png">

##	OpenVINO Accuracy Checker Tool

###	Installation
Follow the guide of [openvinotookit_open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker).  

```
$ cd /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/accuracy_checker   
$ sudo apt-get install python3 python3-dev python3-setuptools python3-pip  
$ python3 setup.py install
```

###	Modify
Copy the contents of **tools/accuracy_checker** to accuracy checker tool path you installed before.  

Added content:  
1. An adapter specifically used for pytorch yolo models: pytorch_yolo.  
2. Non-maximum suppression method used in yolov4: diou_nms.  

```
$ cp -r tools/accuracy_checker /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/
```

###	Download Datasets

```
$ wget http://images.cocodataset.org/zips/val2017.zip  
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip  
$ unzip -d annotations_trainval2017/ annotations_trainval2017.zip    
$ unzip -d annotations_trainval2017/annotations/ val2017.zip  
```

###	Run Accuracy Checker Tool 
Same as before, take yolov5s as an example. Use the command below to run accuracy checker.  

```
$ accuracy_check -c configs/accuracy-check-yolov5.yml -m . --definitions /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/accuracy_checker/dataset_definitions.yml -s annotations_trainval2017/annotations/ -td CPU  
```

##	References

YOLOV3  
https://github.com/zldrobit/onnx_tflite_yolov3  

YOLOV4  
https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u5_preview  
https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/u5_mish_preview  

Scaled-YOLOV4  
https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-large#installation  
https://github.com/linghu8812/tensorrt_inference/tree/master/ScaledYOLOv4  
https://github.com/DataXujing/ScaledYOLOv4  

YOLOV5  
https://github.com/ultralytics/yolov5  
https://github.com/violet17/yolov5_demo  

OpenVINO  
https://docs.openvinotoolkit.org/  
https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker  
