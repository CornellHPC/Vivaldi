# Distributed Popcorn

Install CombBLAS (our local version, which has some bugfixes for which we should ultimately raise a PR to CombBLAS)
```bash
git clone git@github.com:nakuliyer/CombBLAS.git
cd CombBLAS
git switch combblas-gpu
mkdir _build && mkdir _install
cd _build
cmake -DCMAKE_INSTALL_PREFIX=../_install ..
cmake --build . --target install
cd ../_install
export COMBBLAS_INSTALL=$(pwd)
```

Install SLATE
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

Test SLATE (optional)
```
cd slate
make check
echo "srun --nodes=4 --ntasks=16 --cpus-per-task=8 ./test/tester gemm" > job.sh
sbatch job.sh
```

Build source
```bash
git clone git@github.com:mattrrubino/distributed-popcorn.git
cd distributed-popcorn
git switch matt-test
mkdir build && cd build

# Host memory
cmake -DCOMBBLASS_INSTALL=$COMBBLAS_INSTALL -DSLATE_INSTALL=$SLATE_INSTALL ..
# Device memory
cmake -DGPU=true -DCOMBBLASS_INSTALL=$COMBBLAS_INSTALL -DSLATE_INSTALL=$SLATE_INSTALL ..

cmake --build .
```

Run source
```
sbatch job.sh
```

Enable CUDA-aware MPI
```
export SLATE_GPU_AWARE_MPI=1
export MPICH_GPU_SUPPORT_ENABLED=1
```

## Notes

Make sure to undefine the macro `Error` after including CombBLAS and before including SLATE
to prevent the macro name collision (will emit preprocessing error otherwise).

Make sure to use the standard namespace before including CombBLAS.

Make sure to run on GPUs since CombBLAS SpMM requires cuSparse.

Make sure to clean everything up before calling `MPI_Finalize`.

Define `COMBBLAS_DEBUG` to get extra debugging information printed by CombBLAS.

CombBLAS seems to only support sparse matrix left multiply dense matrix, not right multiply.
This is problematic since there is no transposition operation defined for distributed dense matrices.

