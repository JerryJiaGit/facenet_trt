# facenet_trt
NVIDIA TensorRT implementation for facenet with pre-train SavedModel.
facenet is a project from https://github.com/davidsandberg/facenet to do face recognition with tensorflow.

# Changes 
1. facenet.py: Enable facenet pre-train SavedModel with TRT
2. face.py: Add threshold of probobility for return, change minimum size of face to 50px, change gpu_memory_fraction to 0.4 
3. /align/detect_face.py: Enable TensorRT for PNET, RNET and ONET graph

# TensorRT and setup
TensorRT introduction:
https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/
https://docs.nvidia.com/deeplearning/dgx/integrate-tf-trt/index.html

Setup:
HW: GV100 PCIE graphic and intel x86_64
OS: Ubuntu 16.04
Driver: 384.111
CUDA: 9.0.179
cuDNN:7.3.1.20
TensorRT:4.0.1.6
TensorFlow: tensorflow-gpu 1.12

# Result
Face detection with MTCNN: test 30 times with different image at different resolution

-original network: avg 41.948318 ms

-tensorrt network FP32 : avg 42.783976 ms

-tensorrt network FP16 : avg 42.028268 ms

Face indentify with Inception-ResNet-v1
: test 27 times with different image (crop and alignment 160x160)

-original network: avg 13.713258 ms	

-tensorrt network FP32 : avg 	11.296281	ms

-tensorrt network FP16 : avg 10.54711 ms
