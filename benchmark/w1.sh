#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=sl_out/w1

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/device_wrapper $PWD/../build/main"
export DATA="-i $PWD/../data/rand -m 70000 -n 64 --niter 100 -o $PWD/assignments/w1"
export CLUSTERS="-k 128"

echo "Running weak scaling test on 1 rank!"
echo ""

for i in {1..20}; do
  echo "Trial $i"
  srun -N 1 --ntasks-per-node 1 --cpus-per-task 32 --cpu-bind cores -G 1 $EXE_PATH $DATA $CLUSTERS --basic --benchmark $PWD/basic_time/w1_$i
  echo ""
done

echo "Done!"

