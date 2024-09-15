#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --constraint=cpu
#SBATCH --qos=debug

cd /global/homes/m/mrrubino/cpp/distributed-popcorn/build

# Run 16 MPI processes with 8 physical cores per process (8*2=16 logical cores)
srun --ntasks 16 --cpus-per-task 16 main
