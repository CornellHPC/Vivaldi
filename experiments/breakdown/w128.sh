#!/bin/bash
#SBATCH --nodes=32
#SBATCH --gpus=128
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=sl_out/w128

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/device_wrapper $PWD/../build/main"
export DATA="-i $PWD/../data/rand -m 791959 -n 64 --niter 100 -o $PWD/assignments/w128"
export CLUSTERS="-k 128"
export TRIALS=5

module load cudatoolkit/12.2

echo "Running weak scaling test on 128 ranks!"
echo ""

for i in $(seq 1 $TRIALS); do
  echo "Full trial $i"
  srun -N 32 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 128 $EXE_PATH $DATA $CLUSTERS --benchmark $PWD/breakdown_time/w128_$i
  echo ""
done

echo "Done!"
