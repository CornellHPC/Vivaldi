#!/bin/bash
export DVS_MAXNODES=1__
# export MPICH_GPU_SUPPORT_ENABLED=1
# export SLATE_GPU_AWARE_MPI=1
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

EXE_PATH=build/main
DATA_PATH=$PWD/data/test
CLUSTERS=8

# Run 16 MPI processes
# srun -n 16 -c 32 --cpu_bind=cores -G 16 $EXE_PATH $DATA_PATH 4 4 $CLUSTERS
# srun -n 16 -c 32 --cpu_bind=cores --gpus-per-node=1 -G 16 $EXE_PATH $DATA_PATH 128 128 $CLUSTERS
srun --nodes=4 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpu-bind=single:1 $EXE_PATH $DATA_PATH 128 128 $CLUSTERS
