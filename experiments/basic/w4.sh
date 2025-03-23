#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:20:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output=sl_out/w4

export DVS_MAXNODES=1__
export EXE_PATH="$PWD/../build/device_wrapper $PWD/../build/main"
export DATA="-i $PWD/../data/rand -m 140000 -n 64 --niter 100 -o $PWD/assignments/w4"
export CLUSTERS="-k 128"
export TRIALS=5

module load cudatoolkit/12.2

echo "Running weak scaling test on 4 ranks!"
echo ""

for i in $(seq 1 $TRIALS); do
  echo "Basic trial $i"
  srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 $EXE_PATH $DATA $CLUSTERS  --benchmark $PWD/basic_time/w4_$i
  echo ""
done

echo "Done!"

