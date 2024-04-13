# CNN for MNIST via Cuda C++

_No Python, No torch, No external wrappers or libraries. Just pure math and pointers_


To compile
```bash
nvcc main.cu -o main -ccbin "[your directory if necessary]"

e.g. nvcc main.cu -o main -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
```
To run
```bash
./main {device}

device is either cpu or gpu
```