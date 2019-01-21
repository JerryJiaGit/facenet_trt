# facenet_trt
NVIDIA TensorRT implementation for facenet with pre-train SavedModel.
facenet is a project from https://github.com/davidsandberg/facenet to do face recognition with tensorflow.

# Changes 
1. facenet.py: Enable facenet pre-train SavedModel with TRT
2. face.py: Add threshold of probobility for return, change minimum size of face to 50px, change gpu_memory_fraction to 0.3 
3. /align/detect_face.py: Enable TensorRT for PNET, RNET and ONET graph
4. face.py and facenet.py: Minor change to support multi-thread
5. face.py: Change input:0 to batch_join:0 to support both TensorRT4 and TensorRT5

# TensorRT introduction
"NVIDIA announced the integration of our TensorRT inference optimization tool with TensorFlow. TensorRT integration will be available for use in the TensorFlow 1.7 branch. TensorFlow remains the most popular deep learning framework today while NVIDIA TensorRT speeds up deep learning inference through optimizations and high-performance runtimes for GPU-based platforms. We wish to give TensorFlow users the highest inference performance possible along with a near transparent workflow using TensorRT. The new integration provides a simple API which applies powerful FP16 and INT8 optimizations using TensorRT from within TensorFlow. TensorRT sped up TensorFlow inference by 8x for low latency runs of the ResNet-50 benchmark." - from NVIDIA website. 

Latest TensorRT version is 5.0.4.

See details from below links:

https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/

https://docs.nvidia.com/deeplearning/dgx/integrate-tf-trt/index.html

See documents for support matrix: https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html

# Usage
Download facenet.py, face.py (optional), /align/detect_face.py and replace original files.

# Setup
HW: GV100 PCIE graphic and intel x86_64

OS: Ubuntu 16.04

Driver: 384.111

CUDA: 9.0.179

cuDNN:7.3.1.20

TensorRT:4.0.1.6

TensorFlow: tensorflow-gpu 1.12


Code also works on Jetson Xavier

HW: Xavier with internal GPU

OS: Ubuntu 18.04

Driver: L4T 4.1.1

CUDA: 10.0.117

cuDNN:7.3.1

TensorRT:5.0.3

TensorFlow: tensorflow-gpu 1.12


HW: x86 with Quadro V100 GPU

OS: Ubuntu 18.04

Driver: 410.93

CUDA: 10.0.117

cuDNN:7.3.1

TensorRT:5.0.3

TensorFlow: tensorflow-gpu 1.12


# Result
TensorRT 4 result

Face detection with MTCNN: test 30 times with different image at different resolution

| Detect Network      | Avg Time |
|-----------------|--------------|
| original network | 41.948318 ms |
| tensorrt network FP32  | 41.948318 ms |
| tensorrt network FP16  | 42.028268 ms |

*Note: suspect MTCNN network is not converted to TensorRT network automatically, will investage more and try plugin later. And also, I found there is no improvement with checkpoints file, so that means we may not get imporvement with similar method for MTCNN graph convert. Suspected this is some bug in TRT, still working on it.

Face identify with Inception-ResNet-v1
: test 27 times with different image (crop and alignment 160x160)

| Identify Network      | Avg Time |
|-----------------|--------------|
| original network | 13.713258 ms |
| tensorrt network FP32  | 11.296281 ms |
| tensorrt network FP16  | 10.54711 ms |

*Note: INT8 not implemented due to some issues which may same as https://github.com/tensorflow/tensorflow/issues/22854
*Note: The result is based on savedmodel file, for checkpoints frozen graph, has no runtime improvement, that may be a bug, still working on it.

TensorRT 5 result
Similar to TRT4 but the runtime improvement with savedmodel is about 11.89% on GV100. 

TensorRT 5 on Xavier result
Similar to TRT4 but the runtime improvement with savedmodel is about 23.15% on Xavier.
