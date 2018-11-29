# facenet_trt
NVIDIA TensorRT implementation for facenet with pre-train SavedModel.
facenet is a project from 

# Changes 
1. facenet.py: Enable facenet pre-train SavedModel with TRT
2. face.py: Use pre-train SavedModel instead of ckpt file and add threshold of probobility for return

# TensorRT and setup
https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/
https://docs.nvidia.com/deeplearning/dgx/integrate-tf-trt/index.html
