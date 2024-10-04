#!/bin/bash
export DVS_MAXNODES=1__
export MPICH_GPU_SUPPORT_ENABLED=1
export SLATE_GPU_AWARE_MPI=1

srun --ntasks 4 --gpus 16 build/main test_new 8 8 3
