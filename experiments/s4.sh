#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=sl_out/s4

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/device_wrapper $PWD/../build/main"
export DATA="-i $PWD/../data/rand -m 70000 -n 64 --niter 100 -o $PWD/assignments/s4"
export CLUSTERS="-k 128"

echo "Running strong scaling test on 4 ranks!"
echo ""

for i in {1..20}; do
  echo "Basic trial $i"
  srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 $EXE_PATH $DATA $CLUSTERS --basic --benchmark $PWD/basic_time/s4_$i
  echo ""
done

for i in {1..20}; do
  echo "Full trial $i"
  srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 $EXE_PATH $DATA $CLUSTERS --benchmark $PWD/breakdown_time/s4_$i
  echo ""
done

echo "Done!"

