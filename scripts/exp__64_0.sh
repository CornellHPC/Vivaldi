#!/bin/bash
#SBATCH --nodes=16
#SBATCH --gpus=64
#SBATCH --time=4:00:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=REPLACE
#SBATCH --output=logs/exp__64_0_out
export DVS_MAXNODES=1__
export EXE_PATH="$PWD/blasbuild/device_wrapper $PWD/blasbuild/main"
module load cudatoolkit/12.2
echo "Running with args $PSCRATCH/poker.t.bin 128000 10 2"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/poker.t.bin 128000 10 2
  echo ""
done

echo "Running with args $PSCRATCH/poker.t.bin 128000 10 5"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/poker.t.bin 128000 10 5
  echo ""
done

echo "Running with args $PSCRATCH/poker.t.bin 128000 10 10"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/poker.t.bin 128000 10 10
  echo ""
done

echo "Running with args $PSCRATCH/poker.t.bin 128000 10 50"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/poker.t.bin 128000 10 50
  echo ""
done

echo "Running with args $PSCRATCH/poker.t.bin 128000 10 100"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/poker.t.bin 128000 10 100
  echo ""
done

echo "Running with args $PSCRATCH/HIGGS.bin 128000 28 2"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/HIGGS.bin 128000 28 2
  echo ""
done

echo "Running with args $PSCRATCH/HIGGS.bin 128000 28 5"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/HIGGS.bin 128000 28 5
  echo ""
done

echo "Running with args $PSCRATCH/HIGGS.bin 128000 28 10"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/HIGGS.bin 128000 28 10
  echo ""
done

echo "Running with args $PSCRATCH/HIGGS.bin 128000 28 50"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/HIGGS.bin 128000 28 50
  echo ""
done

echo "Running with args $PSCRATCH/HIGGS.bin 128000 28 100"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/HIGGS.bin 128000 28 100
  echo ""
done

echo "Running with args $PSCRATCH/mnist8m.scale.bin 128000 784 2"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/mnist8m.scale.bin 128000 784 2
  echo ""
done

echo "Running with args $PSCRATCH/mnist8m.scale.bin 128000 784 5"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/mnist8m.scale.bin 128000 784 5
  echo ""
done

echo "Running with args $PSCRATCH/mnist8m.scale.bin 128000 784 10"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/mnist8m.scale.bin 128000 784 10
  echo ""
done

echo "Running with args $PSCRATCH/mnist8m.scale.bin 128000 784 50"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/mnist8m.scale.bin 128000 784 50
  echo ""
done

echo "Running with args $PSCRATCH/mnist8m.scale.bin 128000 784 100"
echo ""
for i in $(seq 1 5); do
  echo "Trial $i"
  srun -N 16 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 64 $EXE_PATH $PSCRATCH/mnist8m.scale.bin 128000 784 100
  echo ""
done

echo 'Done!'
