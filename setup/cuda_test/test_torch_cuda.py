import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
torch::Tensor add_one_cuda(torch::Tensor x);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__ void add_one_kernel(
    const float* x,
    float* y,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        y[idx] = x[idx] + 1.0f;
    }
}

torch::Tensor add_one_cuda(torch::Tensor x)
{
    auto y = torch::empty_like(x);

    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    add_one_kernel<<<blocks, threads, 0,
        c10::cuda::getCurrentCUDAStream().stream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
"""

print("Compiling CUDA extension...")

module = load_inline(
    name="grad_nav_cuda_test",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["add_one_cuda"],
    with_cuda=True,
    verbose=True,
)

print("Compilation completed.")

x = torch.arange(
    10,
    dtype=torch.float32,
    device="cuda:0"
)

y = module.add_one_cuda(x)

torch.cuda.synchronize()

print("Input:", x)
print("Output:", y)

assert torch.allclose(y, x + 1)

print("CUDA extension test: PASS")
