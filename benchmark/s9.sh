#!/bin/bash
#SBATCH --nodes=3
#SBATCH --gpus=9
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=s9

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/main"
export DATA="$PWD/../data/rand 32000 64"
export CLUSTERS=128

echo "Running strong scaling test on 9 ranks!"
echo ""

for i in {1..5}; do
  echo "Trial $i"
  srun --nodes=3 --ntasks=9 --cpus-per-task=32 --cpu-bind=cores --gpus=9 --gpu-bind=single:1 $EXE_PATH $DATA $CLUSTERS
  echo ""
done

echo "Done!"

