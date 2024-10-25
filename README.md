This repo is simply for training ML models utlilizing YOLO

Additionally, this repo has the added use of Group Normalization by instanling my version of YOLO which contains this as an alternative

Guide:
pip install git+https://github.com/Logan-Cole-Luna/ultralytics.git@Group_Normalization

 It is an alternative to the standard Batch Normalization used by YOLO. This implementation improves YOLO's ability to handle long-tailed datasets and a lower batch count, leading to an overall increase in YOLO accuracy. It performs notably better in handling long-tailed datasets as displayed below, making it relevant to projects that do not have an ideal, uniformly distributed dataset.

Coco128 Dataset, 1000 Epochs, no transfer weights

Batch Normalization:
Class Images Instances Box(P R mAP50 mAP50-95): 100%|██████████| 16/16 [00:01<00:00, 10.17it/s]
all 128 929 0.616 0.472 0.513 0.349
wandb: model/speed_PyTorch(ms) 11.088

Group Normalization:
YOLOv8n summary: 225 layers, 3157200 parameters, 0 gradients, 8.7 GFLOPs
Class Images Instances Box(P R mAP50 mAP50-95): 100%|██████████| 16/16 [00:01<00:00, 12.36it/s]
all 128 929 0.68 0.486 0.549 0.376
wandb: model/speed_PyTorch(ms) 14.852

Another exciting result is that Group Normalization displayed its effectiveness in handling a long-tail distribution dataset such as coco128 which is primarily focused on the person class:

Batch Norm:
Class Images Instances Box(P R mAP50 mAP50-95): 100%|██████████|
all 128 929 0.616 0.472 0.513 0.349
bicycle 3 6 0.749 0.167 0.17 0.136
car 12 46 0.299 0.0563 0.0682 0.0292
motorcycle 4 5 0.661 0.8 0.92 0.748
airplane 5 6 0.32 0.833 0.638 0.419
bus 5 7 0.623 0.714 0.694 0.606
train 3 3 0.355 1 0.913 0.62
truck 5 12 1 0.305 0.479 0.338
boat 2 6 0.175 0.667 0.184 0.1
traffic light 4 14 0.849 0.143 0.15 0.12
stop sign 2 2 1 0 0.503 0.119
bench 5 9 0.217 0.249 0.298 0.154
bird 2 16 0.933 0.688 0.905 0.57
cat 4 4 0.712 0.75 0.825 0.595
dog 9 9 0.205 0.46 0.27 0.166

Group Norm:
Class Images Instances Box(P R mAP50 mAP50-95): 100%|██████████|
bicycle 3 6 0.803 0.167 0.169 0.118
car 12 46 0.695 0.0435 0.0913 0.0522
motorcycle 4 5 0.706 0.8 0.796 0.645
airplane 5 6 0.898 1 0.995 0.7
bus 5 7 0.676 0.429 0.585 0.49
train 3 3 0.759 1 0.995 0.735
truck 5 12 0.825 0.395 0.454 0.313
boat 2 6 0.698 0.396 0.647 0.339
traffic light 4 14 0.756 0.143 0.162 0.125
stop sign 2 2 0.274 0.5 0.498 0.349
bench 5 9 0.426 0.444 0.428 0.269
bird 2 16 0.651 0.625 0.669 0.38
cat 4 4 0.897 1 0.995 0.737
dog 9 9 0.691 0.889 0.828 0.593

An MRE to replicate the results displayed above:

import torch
from ultralytics import YOLO

# Load the model, ensure it has the specification for norm_type: "group", the default currently in yolov8.yaml
# is currently norm_type: "batch" as this is the standard normalization used in YOLO
# Please be sure to change it before running to view changes
model = YOLO("yolov8n.yaml")#.load("yolov8n.pt")  # build from YAML and transfer weights

# Print the normalization type, default to 'batch' if not specified
print(f'\nNormalization type: {model.model.yaml.get("norm_type", "batch")} normalization\n')

results = model.train(data="coco128.yaml", epochs=1000, imgsz=640, batch=4")