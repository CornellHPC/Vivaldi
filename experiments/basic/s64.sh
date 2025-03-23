#!/bin/bash
#SBATCH --nodes=16
#SBATCH --gpus=64
#SBATCH --time=00:20:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=sl_out/s64

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/device_wrapper $PWD/../build/main"
export DATA="-i $PWD/../data/rand -m 70000 -n 64 --niter 100 -o $PWD/assignments/s64"
export CLUSTERS="-k 128"
export TRIALS=5

module load cudatoolkit/12.2

echo "Running strong scaling test on 64 ranks!"
echo ""

for i in $(seq 1 $TRIALS); do
  echo "Basic trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $DATA $CLUSTERS --benchmark $PWD/basic_time/s64_$i
  echo ""
done

echo "Done!"

