# Communication-Avoiding Linear Algebraic Kernel K-Means on GPUs

## Relevant File Tree

```
├── data           - contains small test data
├── experiments    - contains experimentation tools
└── src
    ├── cpop
        ├── cluster.cc/hh        - clustering methods
        ├── compute_kernel.cc/hh - code relevant to computing the kernel matrix
        ├── gpu_kernels.cu/cuh   - CUDA kernels
        ├── utils.cc             - other useful helpers
    ├── CMakeLists.txt           - build configuration
    └── main.cc                  - main algorithm implementation
├── src_combblas   - implementation with CombBLAS
├── tests          - Python serial implementation
└── Makefile       - start point for building and testing
```

## Relevant Library Requirement

This library has been tested with the following requirements:

* CUDA Toolkit 12.2
* GCC 12.3
* [SLATE 2024.10.29](https://github.com/icl-utk-edu/slate/releases/tag/v2024.10.29)
* CombBLAS

### Installing SLATE

The following can be used to install [SLATE](https://github.com/icl-utk-edu/slate). SLATE should be installed in the user’s home directory `~/` (if not, the line `export SLATE_INSTALL := …` in `Makefile` will need to be amended). This will take a while (>30 mins).

```bash
cd ~/
export mpi=cray
export blas=libsci
export CXX=CC
git clone --recursive https://github.com/icl-utk-edu/slate.git
cd slate/blaspp
make -j`nproc`
cd ../lapackpp
make -j`nproc`
cd ..
make -j`nproc`
mkdir _install
make install prefix=_install
cd _install
export SLATE_INSTALL=$(pwd)
```

### Installing CombBLAS
The following can be used to install CombBLAS. CombBLAS should be installed in the user’s home directory `~/` (if not, the line `export COMBBLAS_INSTALL := …` in `Makefile` will need to be amended). This will take a few minutes.
```bash
cd ~/
wget https://zenodo.org/records/15208078/files/CombBLAS-combblas-gpu.zip
unzip CombBLAS-combblas-gpu.zip
mv CombBLAS-combblas-gpu CombBLAS
cd CombBLAS
mkdir _build && mkdir _install
cd _build
cmake -DCMAKE_INSTALL_PREFIX=../_install ..
cmake --build . --target install
cd ../_install
export COMBBLAS_INSTALL=$(pwd)
```

The zip for CombBLAS repo can also be directly downloaded from [here](https://zenodo.org/records/15208078).

## Makefile: Building

Build with `make build`. Relevant options are
* `make build`: build Kettlecorn in `build` directory, to build without fine-grained timing 
(e.g. for benchmarking without breakdown), you may run `make build BASIC=1` as well
* `make blasbuild`: build alternative CombBLAS implementation in `blasbuild` directory

## Makefile: Testing

**Before you proceed, please replace ACCOUNT in Makefile with your account id for your project.**

* `make alloc` requests interactive compute session on your cluster

### Small Dataset Testing

Naive testing on smaller datasets can be done with:
* `make australian` (690 points, 14 features, 2 clusters, Sparse V)
* `make svmguide1` (3089 points, 4 features, 2 clusters, Sparse V)
* `make letter` (15k points, 5k features, 26 clusters, Sparse V)
* `make rand` (70k points, 64 features, 128 clusters, Sparse V)
* You may append `--convergence=1` at the end ofin the Makefile to run with convergence

All of these tests launch their own allocated interactive session. Svmguide1 and Letter request 16 GPUs while the others request 4 GPUs, where 1 node has 4 GPUs. Note that these commands may need to be modified according to the architecture of the cluster you run on. 

More rigorous scaling testing should be done from within the `experiments` folder (for more, see the [README](experiments/README.md) there). Rigorous scaling testing must not use the interactive session and or convergence checking.

* use `--convergence=1` in the Makefile to run with convergence
(ex: build/device_wrapper build/main -i data/letter -m 690 -n 14 -k 2 --convergence=1)
* use `--sparse=0` in the Makefile to run in dense V mode
(ex: build/device_wrapper build/main -i data/letter -m 690 -n 14 -k 2 --sparse=0)
* see `src/cpop/utils.cc` for other runtime arguments

### Additional Profiling
* `make profile` launches nsys profiling on rand on 1 GPU

### Citation

If you find this repo helpful to your work, please cite our article:

```
TBD
```

### Acknowledgement

This research used resources of the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award ASCR-ERCAP0032541. This work was done in collaboration to the Hicrest Laboratory at the University of Trento. 
