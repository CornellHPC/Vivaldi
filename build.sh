cd ~/CombBLAS/_install
export COMBBLAS_INSTALL=$(pwd)
cd ~/slate/_install
export SLATE_INSTALL=$(pwd)
cd ~/distributed-popcorn/build
export mpi=cray
export blas=libsci
export CXX=CC
export SLATE_GPU_AWARE_MPI=1
export MPICH_GPU_SUPPORT_ENABLED=1
export DVS_MAXNODES=1__
export MPICH_GPU_SUPPORT_ENABLED=1
export SLATE_GPU_AWARE_MPI=1
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
cmake -DCOMBBLAS_INSTALL=$COMBBLAS_INSTALL -DSLATE_INSTALL=$SLATE_INSTALL -DGPU=1 ..
cmake --build .
cd ../
# sbatch job.sh
# echo "Repeating command squeue every second"
# while sleep 1; do
#     squeue -u npi2
# done
