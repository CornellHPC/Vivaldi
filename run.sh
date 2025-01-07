#!/bin/bash
export DVS_MAXNODES=1__
# export MPICH_GPU_SUPPORT_ENABLED=1
export SLATE_GPU_AWARE_MPI=0

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

EXE_PATH=build/main
DATA_PATH=$PWD/data/test

# Samples
srun --nodes=1 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=4 --gpu-bind=single:1 $EXE_PATH $PWD/data/small 11 8 4

# Australian
# srun --nodes=1 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=4 --gpu-bind=per_task:1 $EXE_PATH $PWD/data/australian 690 14 2

# Svmguide1
# srun --nodes=4 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=16 --gpu-bind=single:1 $EXE_PATH $PWD/data/svmguide1 3089 4 2

# srun --nodes=4 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=16 --gpu-bind=single:1 build/main+pat data/australian 690 14 2