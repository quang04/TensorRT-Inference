# TensorRT-Inference
# Overview
This project demonstrates how to convert models from other architectures to the TensorRT format to speed up inference time and shows how to run multiple models simultaneously.
The project is currently running inference on 5 models detection and 5 models classification at same time.
# Config
C++20
Cmake>=3.16
1. [Qt 6](https://www.qt.io/download-qt-installer-oss)
2. [Opencv 4.10](https://github.com/opencv/opencv/releases/tag/4.10.0)
   Build with cuda enable
4. [Cuda 11.4 with cudnn 8.6.1](https://developer.nvidia.com/cuda-11-4-0-download-archive)
5. [TensortRT 8.5.3.1](https://developer.nvidia.com/tensorrt-getting-started)
6. [Onnx Runtime 1.12.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.12.1)
   Build with GPU
# Model for tesing
## Make sure your model architecture support tensorRT
+ Classification: [Restnet 50](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet) + Imagenet label
+ Detection: [Yolov8](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes) + COCO label
  
Need convert model to onnx format
# Improvement
The source code is quite heavy on virtual function mechanism
+ Improve by using CRTP approach
+ Improve by using Boost container
# Illustration
![gh1](https://github.com/user-attachments/assets/3719a7d5-1c55-41d7-a2c8-acc3ac9213d4)
Model after convert
![image](https://github.com/user-attachments/assets/cc647ffe-84f4-4450-839d-401ef4548ed1)

![ClassificationResult_modelID0_gpu0](https://github.com/user-attachments/assets/2f6e3ae0-9916-42c5-9ce7-11396ed7093c)
![DetectionResult_modelID0_gpu0](https://github.com/user-attachments/assets/b21389bd-199f-478d-aee1-46b4dfbad27f)

# End
Let me know if you have any suggestion/improvement
