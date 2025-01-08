# Makefile

.PHONY: alloc build env

export SLATE_INSTALL := $(shell cd slate/_install && pwd) # Change slate directory as necessary
export mpi := cray
export blas := libsci
export CXX := CC
export SLATE_GPU_AWARE_MPI := 0
export MPICH_GPU_SUPPORT_ENABLED := 1
export DVS_MAXNODES := 1
export OMP_NUM_THREADS := 1
export OMP_PLACES := threads
export OMP_PROC_BIND := spread

alloc:
	salloc --nodes 4 --qos interactive --time 03:00:00 --constraint gpu --gpus 16 --account m4341

build:
	cd build && \
	cmake -DSLATE_INSTALL=$$SLATE_INSTALL .. && \
	cmake --build . && \
	cd ..