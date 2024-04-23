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

mode is either cpu or gpu_optimized or gpu_naive. If you don't specifiy mode, gpu_optimized is selected as default.
e.g.  
$ ./main gpu_optimized
$ ./main cpu  

gpu_naive mode uses gpu kernels that calculates convolution directly.
gpu_optimized mode uses gpu kernels that calculates convolution via matrix multiplication
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
Maxpool
RELU
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
> [_gpu + convolution via matrix multiplication_]  time : 8.64s  accuracy : 97.94%  
> [_gpu + direct convolution_]  time : 3.76s  accuracy : 97.90%  
> [_cpu_]  time : 1344s
<br>

Benchmark setting 2
```
Batch size = 64
learning rate = 5e-3
Convnet : 
    Conv(1, 256, 5,5)
    Maxpool()
    RELU()
    Conv(256, 256, 3,3)
    Maxpool()
    RELU()
    Conv(256, 10, 5,5)
    Softmax()
```
Training for 1 epoch
> [_gpu + convolution via matrix multiplication_]  time : 228s  accuracy : 97.5%  
> [_gpu + direct convolution_]  time : 262s  accuracy : 97.56%    
> the optimized version works faster for larger(practical) networks since the whole unrolling, transposing, and multiplying widened matrices simply acts as an overhead for small networks. 