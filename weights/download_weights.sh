#!/bin/bash
# Download latest models from https://github.com/ultralytics/yolov5/releases and https://github.com/Chen-MingChang/pytorch_YOLO_OpenVINO_demo/releases
# Usage:
#    $ bash weights/download_weights.sh

python - <<EOF
from utils.google_utils import attempt_download

attempt_download('yolov4-pacsp.pt')
attempt_download('yolov4-pacsp-mish.pt')
attempt_download('yolov4-p5.pt')
attempt_download('yolov4-p6.pt')
attempt_download('yolov4-p7.pt')
for x in ['s', 'm', 'l', 'x']:
    attempt_download(f'yolov5{x}.pt')

EOF
