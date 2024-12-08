#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus=16
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --account=m4341
#SBATCH --output=out/%j

export DVS_MAXNODES=1__
# export MPICH_GPU_SUPPORT_ENABLED=1
# export SLATE_GPU_AWARE_MPI=1
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

EXE_PATH=build/main
DATA="$PWD/data/rand 32000 64"
CLUSTERS=128

# Run 16 MPI processes
srun --nodes=4 --ntasks=16 --cpus-per-task=32 --cpu-bind=cores --gpus=16 --gpu-bind=single:1 \
  nsys profile --stats=true -t cuda,mpi -o benchmark/report \
  $EXE_PATH $DATA $CLUSTERS

