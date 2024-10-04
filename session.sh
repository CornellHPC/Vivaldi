#!/bin/bash
salloc --nodes 4 --qos interactive --time 01:00:00 --constraint gpu --gpus 16 --account m4341

export DVS_MAXNODES=1__
export MPICH_GPU_SUPPORT_ENABLED=1
export SLATE_GPU_AWARE_MPI=1
