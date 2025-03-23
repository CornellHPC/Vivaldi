#!/bin/bash
#SBATCH --nodes=8
#SBATCH --gpus=32
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=sl_out/w32

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/device_wrapper $PWD/../build/main"
export DATA="-i $PWD/../data/rand -m 395968 -n 64 --niter 100 -o $PWD/assignments/w32"
export CLUSTERS="-k 128"
export TRIALS=5

module load cudatoolkit/12.2

echo "Running weak scaling test on 32 ranks!"
echo ""

for i in $(seq 1 $TRIALS); do
  echo "Basic trial $i"
  srun -N 8 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 32 $EXE_PATH $DATA $CLUSTERS --benchmark $PWD/basic_time/w32_$i
  echo ""
done

echo "Done!"
