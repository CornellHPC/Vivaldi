#!/bin/bash
#SBATCH --nodes=4
#SBATCH --time=00:00:10
#SBATCH --constraint=cpu
#SBATCH --qos=debug

cd /global/homes/m/mrrubino/cpp/distributed-popcorn/build

# Run 4 MPI processes
srun --ntasks 4 main
