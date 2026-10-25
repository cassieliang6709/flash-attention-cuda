# Build:  TORCH_CUDA_ARCH_LIST="8.9" pip install -e . --no-build-isolation
#   8.9 = RTX 4090 (Ada), 9.0 = H100 (Hopper), 8.0 = A100.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash_cuda",
    version="0.1.0",
    packages=["flash_cuda"],
    ext_modules=[
        CUDAExtension(
            name="flash_cuda._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/naive.cu",
                "csrc/fa1.cu",
                "csrc/fa2.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                # -lineinfo: lets Nsight Compute map stalls back to source lines.
                # No --use_fast_math: we want expf() accuracy for the FP32 tests.
                "nvcc": ["-O3", "-std=c++17", "-lineinfo", "--ptxas-options=-v"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
