#!/bin/bash
#SBATCH --nodes=9
#SBATCH --gpus=36
#SBATCH --time=01:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=w36

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/main"
export DATA="$PWD/../data/rand 96000 64"
export CLUSTERS=128

echo "Running weak scaling test on 36 ranks!"
echo ""

for i in {1..5}; do
  echo "Trial $i"
  srun --nodes=9 --ntasks=36 --cpus-per-task=32 --cpu-bind=cores --gpus=36 --gpu-bind=single:1 $EXE_PATH $DATA $CLUSTERS
  echo ""
done

echo "Done!"
