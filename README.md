# Distributed Popcorn

Install CombBLAS
```bash
git clone git@github.com:PASSIONLab/CombBLAS.git
cd CombBLAS
git switch combblas-gpu
wget -O arr-setter.patch https://raw.githubusercontent.com/mattrrubino/distributed-popcorn/refs/heads/matt-test/arr-setter.patch?token=GHSAT0AAAAAACTEC2LG3Z5PZYVNZ3M6PBNYZXPJTWQ
git apply arr-setter.patch
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
cmake -DCOMBBLASS_INSTALL=$COMBBLAS_INSTALL -DSLATE_INSTALL=$SLATE_INSTALL ..
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

