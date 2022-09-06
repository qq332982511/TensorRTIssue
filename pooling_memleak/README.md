TensorRT 7.2.2.3 memleak when use `setDeviceMemory` API  
ref https://github.com/NVIDIA/TensorRT/issues/2290  
```
card: T4
g++ (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
```
reproduce step:  
```bash
cd TensorRTIssue/pooling_memleak/
mkdir build
cd build
python3 ../dump_network.py
cmake .. -GNinja -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda_dir/cuda-11.4/ -DBUILD_WITH_ASAN=ON -DTRT_ROOT_DIR=/usr/local/tensorrt_dir/TensorRT-7.2.2.3/ -DCUDNN_ROOT_DIR=/usr/local/cudnn_dir/cudnn-v8.2.1 
ninja
export LD_LIBRARY_PATH=/usr/local/cuda_dir/cuda-11.4/lib64:/usr/local/cudnn_dir/cudnn-v8.2.1/lib64:/usr/local/tensorrt_dir/TensorRT-7.2.2.3/lib:/usr/local/cuda_dir/cuda-11.1/lib64
ASAN_OPTIONS=protect_shadow_gap=0 ./test
```
you will see asan report memleak like
```
==781098==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 32292 byte(s) in 299 object(s) allocated from:
    #0 0x7f6305b84448 in operator new(unsigned long) (/usr/lib/x86_64-linux-gnu/libasan.so.4+0xe0448)
    #1 0x7f626870d500 in cudnnCreatePoolingDescriptor (/usr/local/cudnn_dir/cudnn-v8.2.1/lib64/libcudnn_ops_infer.so.8+0x259500)
    #2 0x7f62e118e315 in nvinfer1::rt::SafeExecutionContext::allocateResourcesForLayers(int, int, CUstream_st*) (/usr/local/tensorrt_dir/TensorRT-7.2.2.3/lib/libnvinfer.so.7+0xe3a315)

SUMMARY: AddressSanitizer: 32292 byte(s) leaked in 299 allocation(s).
```