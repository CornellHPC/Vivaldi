#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus=16
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=s16

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/device_wrapper $PWD/../build/main"
export DATA="$PWD/../data/rand 46000 64"
export CLUSTERS=128

echo "Running strong scaling test on 16 ranks!"
echo ""

for i in {1..5}; do
  echo "Trial $i"
  srun -N 4 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 16 $EXE_PATH $DATA $CLUSTERS
  echo ""
done

echo "Done!"

