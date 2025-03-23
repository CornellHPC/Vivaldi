#!/bin/bash
#SBATCH --nodes=64
#SBATCH --gpus=256
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=sl_out/w256

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/device_wrapper $PWD/../build/main"
export DATA="-i $PWD/../data/rand -m 1120000 -n 64 --niter 100 -o $PWD/assignments/w256"
export CLUSTERS="-k 128"
export TRIALS=5

module load cudatoolkit/12.2

echo "Running weak scaling test on 256 ranks!"
echo ""

for i in $(seq 1 $TRIALS); do
  echo "Basic trial $i"
  srun -N 64 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 256 $EXE_PATH $DATA $CLUSTERS --benchmark $PWD/basic_time/w256_$i
  echo ""
done

echo "Done!"
