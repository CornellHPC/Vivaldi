# Distributed Popcorn: Multi-GPU Kernel K-Means with Sparse Linear Algebra

## File Tree
```
├── CMakeLists.txt - build configs
├── data - contains test data
├── job.sh - launching Perlmutter jobs
├── Makefile
├── README.md
├── run.sh
├── src
│   ├── main.cc - main algorithm implementation
│   └── popcorn
│       ├── compute_kernel.cc - helpers for computing the kernel matrix
│       ├── compute_sparse.cc - helpers for all things related to sparse math
│       ├── gpu_kernels.cu - cuda kernels
│       ├── utils.cc - other useful helpers
└── tests
```

## Installation

Install [SLATE](https://github.com/icl-utk-edu/slate).
This will take a while (~30 mins).

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

Testing SLATE (Optional)
```
cd slate
make check
echo "srun --nodes=4 --ntasks=16 --cpus-per-task=8 ./test/tester gemm" > job.sh
sbatch job.sh
```

## Makefile
- `make alloc`: allocating interactive session on Perlmutter
- `make build`: building source (run `mkdir build` before if no build directory)

## Other

Launching job
```
sbatch job.sh
```

Enable CUDA-aware MPI
```
export SLATE_GPU_AWARE_MPI=1
export MPICH_GPU_SUPPORT_ENABLED=1
```

## Notes
- Make sure to clean everything up before calling `MPI_Finalize`.