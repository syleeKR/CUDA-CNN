# CNN for MNIST via Cuda C++

_No Python, No torch, No tensors_  
_No external frameworks, libraries, and dependencies_  
_Just pure C++/CUDA with the help of simple math!_

## QuickStart
To compile
```bash
$ nvcc main.cu -o main -ccbin "[your directory if necessary]"

e.g.
$ nvcc main.cu -o main -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
```
To run
```bash
$ ./main {device}

device is either cpu or gpu
e.g.  
$ ./main gpu
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
Checked that training for 6 epochs gives us 95% accuracy, and that training with naive gpu kernel gives us 35s/epoch

>_TODO : Can definitely give us speed up_  
    -     Tiling  
    -     Formulating forward/backward algorithm into matrix multiplication

