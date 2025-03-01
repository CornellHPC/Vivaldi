#!/bin/bash
#SBATCH --nodes=16
#SBATCH --gpus=64
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=sl_out/w64

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/device_wrapper $PWD/../build/main"
export DATA="-i $PWD/../data/rand -m 460000 -n 64 --niter 100 -o $PWD/assignments/w64"
export CLUSTERS="-k 128"
export TRIALS=5

module load cudatoolkit/12.2

echo "Running weak scaling test on 64 ranks!"
echo ""

for i in $(seq 1 $TRIALS); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $DATA $CLUSTERS --basic --benchmark $PWD/basic_time/w64_$i
  echo ""
done

for i in $(seq 1 $TRIALS); do
  echo "Full trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $DATA $CLUSTERS --benchmark $PWD/breakdown_time/w64_$i
  echo ""
done

echo "Done!"
