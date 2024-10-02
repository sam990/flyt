export CUDA_NVCC_FLAGS="-cudart=shared "
export PYTORCH_CUDA_ALLOC_CONF="backend:native"
export TORCH_CUDA_ARCH_LIST="8.6"
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export EXTRA_NVCCFLAGS="-cudart=shared"
export NVCC_APPEND_FLAGS='-cudart=shared'
export LIBRARY_PATH='/usr/local/cuda-12.3/lib64/'
export BUILD_TEST="OFF"
export USE_CUDNN="OFF"
export USE_DISTRIBUTED="OFF"
export USE_NCCL="OFF"
export USE_CUDA="ON"