# CNN for MNIST via Cuda C++

_No Python, No torch, No tensors_  
_No external frameworks, libraries, and dependencies_  
_Just pure C++/CUDA with some simple math!_

## QuickStart
To compile
```bash
$ nvcc main.cu -o main -ccbin "[your directory if necessary]" -lcublas

e.g.
$ nvcc main.cu -o main -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64" -lcublas
```
To run
```bash
$ ./main {mode}

mode is either cpu or gpu or gpu_mm
e.g.  
$ ./main gpu
$ ./main gpu_mm  

gpu mode uses gpu kernels that calculates convolution directly.
gpu_mm mode uses gpu kernels that calculates convolution via matrix multiplication
```

## Notes
The following layers are implemented
```bash
Conv 
Maxpool 
RELU 
Softmax
```
The gpu speed up is devoted to the following layers
```bash
Conv
```

### Results
---
Benchmark setting 1
```
Batch size = 64
learning rate = 5e-3
Convnet : 
    Conv(1, 16, 5,5)
    Maxpool()
    RELU()
    Conv(16, 32, 3,3)
    Maxpool()
    RELU()
    Conv(32, 10, 5,5)
    Softmax()
```
Training for 1 epoch
> [_gpu + convolution via matrix multiplication_]  time : 75.76s  accuracy : 97.76%  
> [_gpu + direct convolution_]  time : 69.27  accuracy : 97.76%  
> [_cpu_]  
<br>

Benchmark setting 2
```
Batch size = 64
learning rate = 5e-3
Convnet : 
    Conv(1, 64, 5,5)
    Maxpool()
    RELU()
    Conv(64, 256, 3,3)
    Maxpool()
    RELU()
    Conv(256, 10, 5,5)
    Softmax()
```
Training for 1 epoch
> [_gpu + convolution via matrix multiplication_]  time : 348.78s  accuracy : 97.68%  
> [_gpu + direct convolution_]  time : 350.51s  accuracy : 97.74%  
> [_cpu_]  
