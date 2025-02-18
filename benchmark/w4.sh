#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=w4

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/device_wrapper $PWD/../build/main"
export DATA="$PWD/../data/rand 64000 64"
export CLUSTERS=128

echo "Running weak scaling test on 4 ranks!"
echo ""

for i in {1..5}; do
  echo "Trial $i"
  srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 $EXE_PATH $DATA $CLUSTERS
  echo ""
done

echo "Done!"

