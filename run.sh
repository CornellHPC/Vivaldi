#!/bin/bash
export DVS_MAXNODES=1__
export MPICH_GPU_SUPPORT_ENABLED=1
export SLATE_GPU_AWARE_MPI=1
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

EXE_PATH=build/main
POINT_PATH=$PWD/data/test
CLUSTERS=3

# Run 16 MPI processes
# srun -n 16 -c 32 --cpu_bind=cores -G 16 --gpu-bind=single:1 $EXE_PATH test 4 4 $CLUSTERS
srun -n 16 -c 32 --cpu_bind=cores -G 16 --gpu-bind=single:1 $EXE_PATH POINT_PATH 8 8 $CLUSTERS
