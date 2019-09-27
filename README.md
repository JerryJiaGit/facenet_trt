# facenet_trt
NVIDIA TensorRT implementation for facenet with pre-train SavedModel.
facenet is a project from https://github.com/davidsandberg/facenet to do face recognition with tensorflow.

# Changes 
1. facenet.py: Enable facenet pre-train SavedModel with TRT
2. face.py: Add threshold of probobility for return, change minimum size of face to 50px, change gpu_memory_fraction to 0.3 
3. /align/detect_face.py: Enable TensorRT for PNET only, keep RNET and ONET graph same as before due to batch size warning
4. face.py and facenet.py: Minor change to support multi-thread
5. face.py: Change input:0 to batch_join:0 to support both TensorRT4 and TensorRT5
6. face.py: Add process for TRT INT8 calib if INT8ENABLE=True

# TensorRT introduction
"NVIDIA announced the integration of our TensorRT inference optimization tool with TensorFlow. TensorRT integration will be available for use in the TensorFlow 1.7 branch. TensorFlow remains the most popular deep learning framework today while NVIDIA TensorRT speeds up deep learning inference through optimizations and high-performance runtimes for GPU-based platforms. We wish to give TensorFlow users the highest inference performance possible along with a near transparent workflow using TensorRT. The new integration provides a simple API which applies powerful FP16 and INT8 optimizations using TensorRT from within TensorFlow. TensorRT sped up TensorFlow inference by 8x for low latency runs of the ResNet-50 benchmark." - from NVIDIA website. 

Latest TensorRT version is 5.0.4.

See details from below links:

https://devblogs.nvidia.com/tensorrt-integration-speeds-tensorflow-inference/

https://docs.nvidia.com/deeplearning/dgx/integrate-tf-trt/index.html

See documents for support matrix: https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html

TRT Installation document: https://developer.download.nvidia.com/compute/machine-learning/tensorrt/docs/5.0/GA_5.0.2.6/TensorRT-Installation-Guide.pdf

# Usage
1. Get GPU cuda/cudnn, tensorflow-gpu and TensorRT ready
2. Get facenet ready with https://github.com/davidsandberg/facenet ready
3. Download here facenet.py, face.py (optional), /align/detect_face.py and replace original files.

# Setup

| HW | Ubuntu | Driver | CUDA | cuDNN | TensorRT | TensorFlow |
|-------------------------------------|--------|----------|---------|----------|---------|--------|
| Tesla V100 graphic and intel x86_64 | 16.04 | 384.111 | 9.0.179 | 7.3.1 | 4.0.1.6 | 1.12 gpu |
| Quadro V100 graphic and intel x86_64 | 18.04 | 410.93 | 10.0.117 | 7.3.1 | 5.0.3 | 1.12 gpu |
| Jetson Xavier with internal GV10B GPU | 18.04 | L4T 4.1.1 | 10.0.117 | 7.3.1| 5.0.3 | 1.12 gpu |

# Result

![](TRT_Runtime_Compare_Result.png?raw=true)

*Note: this table is only for face identify inception-resnet v1 network savedmodel runtime improvement compare. Xavier is Jetson Xavier with L4T 4.1.1.

TensorRT 4 result

Face detection with MTCNN: test 30 times with different image at different resolution

| Detect Network      | Avg Time |
|------------------------|------------------------|
| original network ckpt | 41.948318 ms |
| tensorrt network FP32  | 41.948318 ms |
| tensorrt network FP16  | 42.028268 ms |

*Note: suspect MTCNN network is not converted to TensorRT network automatically, will investage more and try plugin later. And due to batch mis-match warning, only enabled pnet TRT convert right now.

Face identify with Inception-ResNet-v1
: test 27 times with different image (crop and alignment 160x160)

| Identify Network      | Avg Time |
|------------------------|------------------------|
| original network ckpt | 13.713258 ms |
| tensorrt network FP32  | 11.296281 ms |
| tensorrt network FP16  | 10.54711 ms |

*Note: INT8 not implemented due to calib issues "nvinfer1::DimsCHW nvinfer1::getCHW(const nvinfer1::Dims): Assertion `d.nbDims >= 3' failed", it is caused by TRT4, with new TRT5, there is no such problem, but still have other issues, see "issues" for more detailed.

*Note: The result is based on savedmodel file, for checkpoints frozen graph, it has similar result.

TensorRT 5 result
Similar to TRT4 but the runtime improvement with savedmodel is about 11.89% on GV100. 

TensorRT 5 on Xavier result
Similar to TRT4 but the runtime improvement with savedmodel is about 23.15% on Xavier: test 20 times with same image (crop and alignment 160x160, except of first long init one)

| Identify Network      | Avg Time |
|-----------------------------------|-----------------------------------|
| original network ckpt | 45.034961 ms |
| tensorrt network savedmodel FP16  | 37.567716 ms |


