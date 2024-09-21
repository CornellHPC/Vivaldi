#!/bin/bash
#SBATCH --nodes=4
#SBATCH --time=00:00:10
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --output=out/%j

export DVS_MAXNODES=1__

EXE_PATH=build/main
POINT_PATH=$PWD/test
CLUSTERS=3

# Run 4 MPI processes
srun --ntasks 4 $EXE_PATH $POINT_PATH $CLUSTERS
