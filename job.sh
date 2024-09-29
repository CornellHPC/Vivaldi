#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus=16
#SBATCH --time=00:00:10
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --account=m4341
#SBATCH --output=out/%j

export DVS_MAXNODES=1__
export MPICH_GPU_SUPPORT_ENABLED=1
export SLATE_GPU_AWARE_MPI=1

EXE_PATH=build/main
POINT_PATH=$PWD/test
CLUSTERS=3

# Run 4 MPI processes
srun --ntasks 4 --gpus 16 $EXE_PATH $POINT_PATH $CLUSTERS
