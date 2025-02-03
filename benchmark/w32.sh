#!/bin/bash
#SBATCH --nodes=8
#SBATCH --gpus=32
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=w32

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/main"
export DATA="$PWD/../data/rand 90510 64"
export CLUSTERS=128

echo "Running weak scaling test on 32 ranks!"
echo ""

for i in {1..5}; do
  echo "Trial $i"
  srun --nodes=8 --ntasks=32 --cpus-per-task=32 --cpu-bind=cores --gpus=32 --gpu-bind=single:1 $EXE_PATH $DATA $CLUSTERS
  echo ""
done

echo "Done!"
