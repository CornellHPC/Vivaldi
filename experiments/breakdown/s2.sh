#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=00:20:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=sl_out/s2

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/device_wrapper $PWD/../build/main"
export DATA="-i $PWD/../data/rand -m 70000 -n 64 --niter 100 -o $PWD/assignments/s2"
export CLUSTERS="-k 128"
export TRIALS=5

module load cudatoolkit/12.2

echo "Running strong scaling test on 2 ranks!"
echo ""

for i in $(seq 1 $TRIALS); do
  echo "Full trial $i"
  srun -N 1 --ntasks-per-node 2 --cpus-per-task 32 --cpu-bind cores -G 2 $EXE_PATH $DATA $CLUSTERS --benchmark $PWD/breakdown_time/s2_$i
  echo ""
done

echo "Done!"

