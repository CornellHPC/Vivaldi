cd slate/_install  # change as necessary
export SLATE_INSTALL=$(pwd)

cd ../../build  # change as necessary
export mpi=cray
export blas=libsci
export CXX=CC
export SLATE_GPU_AWARE_MPI=1
export MPICH_GPU_SUPPORT_ENABLED=1
export DVS_MAXNODES=1__
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
cmake -DSLATE_INSTALL=$SLATE_INSTALL ..
cmake --build .
cd ..