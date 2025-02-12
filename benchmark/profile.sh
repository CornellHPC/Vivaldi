#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:01:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=profile

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/main"
export DATA="$PWD/../data/rand 46000 64"
export CLUSTERS=128

echo "Running nsys on 1 rank!"
echo ""

srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --cpu-bind=cores --gpus=1 --gpu-bind=single:1 \
  nsys profile --stats=true --cuda-memory-usage=true --trace=cuda,cublas,cusparse,mpi --output=/tmp/report \
  $EXE_PATH $DATA $CLUSTERS

echo "Done!"

