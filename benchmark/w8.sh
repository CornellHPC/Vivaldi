#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --time=00:15:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=w8

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/main"
export DATA="$PWD/../data/rand 45255 64"
export CLUSTERS=128

echo "Running weak scaling test on 8 ranks!"
echo ""

for i in {1..5}; do
  echo "Trial $i"
  srun --nodes=2 --ntasks=8 --cpus-per-task=32 --cpu-bind=cores --gpus=8 --gpu-bind=single:1 $EXE_PATH $DATA $CLUSTERS
  echo ""
done

echo "Done!"

