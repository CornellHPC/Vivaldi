# ClusterPop: Multi-GPU Kernel K-Means with Sparse Linear Algebra

## File Tree

```
├── CMakeLists.txt - build configs
├── Makefile       - start point for building and testing
├── README.md
├── run.sh
├── data           - contains test data
├── experiments    - contains experimentation tools, see there for more
├── tests          - correctness testing tools
└── src
    ├── main.cc                  - main algorithm implementation
    ├── test.cc                  - algorithm unit-testing
    └── cpop
        ├── cluster.cc/hh        - clustering methods
        ├── compute_kernel.cc/hh - computing the kernel matrix
        ├── gpu_kernels.cu/cuh   - cuda kernels
        ├── utils.cc             - other useful helpers
```

## Relevant Library Requirements

This library has been tested with the following requirements:

* CUDA Toolkit 12.2
* GCC 12.3
* [SLATE 2024.10.29](https://github.com/icl-utk-edu/slate/releases/tag/v2024.10.29)

### Installing SLATE

The following can be used to insall [SLATE](https://github.com/icl-utk-edu/slate). SLATE should be installed in the user’s home directory (if not, the line `export SLATE_INSTALL := …` in `Makefile` will need to be amended). This will take a while (\~30 mins):

```bash
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

Testing SLATE (Optional):

```bash
cd slate
make check
echo "srun --nodes=4 --ntasks=16 --cpus-per-task=8 ./test/tester gemm" > job.sh
sbatch job.sh
```

## Building

Build with `make build`. Relevant options are

* `make build BASIC=1`: build without fine-grained timing, e.g. for benchmarking without breakdown

## Testing

### Unit-Testing

Units tests can be run with `make test`.

### Dataset Testing

Naïve dataset testing can be done with

* `make small` (11 points, 8 features, 2 clusters, Sparse V)
* `make australian` (690 points, 14 features, 2 clusters, Sparse V)
* `make svmguide1` (3089 points, 4 features, 2 clusters, Sparse V)
* `make letter` (15k points, 5k features, 26 clusters, Sparse V)
* `make rand` (70k points, 64 features, 128 clusters, Sparse V)
* `make profile` (70k points, 64 features, 128 clusters, Sparse V)


All of these tests launch their own allocated interactive session. Svmguide1 and Letter request 16 GPUs (4 nodes) while the other tests request 4 GPUs (1 node). More rigorous scaling testing should be done from within the `experiments` folder (for more, see the README there). Rigorous scaling testing must not use the interactive session and or convergence checking.

* use `--convergence=1` in the Makefile to run with convergence
* use `--sparse=0` in the Makefile to run in dense V mode
* see `utils.cc` for other runtime arguments

### Correctness Testing

Correctness testing can be done with `make compare`. Todo


