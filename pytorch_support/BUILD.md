> Tested on pytorch version 2.3.0

## Building Steps

1. Clone/download source code for pytorch 2.3.0
```bash
git clone --recursive "https://github.com/pytorch/pytorch"
cd pytorch
git checkout v2.3.0
PYTORCH_CODE=`pwd`
```
2. Create/activate a conda environment for pytorch and dependency installation.
3. Download dependencies: 
```bash
conda install cmake ninja
pip install -r "$PYTORCH_CODE"/requirements.txt
conda install intel::mkl-static intel::mkl-include
conda install -c pytorch magma-cuda121 
```
3. Configure pytorch build and install
```bash
./build_pytorch.sh
```

4. In case pytorch build is terminated due to out of memory issue restart the build. First load the environment variables `source build_env.sh` in flyt/pytorch_support directory. Then start the build again by running `python3 setup.py install` in the pytorch directory.


