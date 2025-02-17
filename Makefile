# Makefile

.PHONY: build alloc env small australian svmguide1 letter

export SLATE_INSTALL := $(shell cd ~/slate/_install && pwd) # Change slate directory as necessary
export mpi := cray
export blas := libsci
export CXX := CC
export DVS_MAXNODES := 1__
export SLATE_GPU_AWARE_MPI := 1
export MPICH_GPU_SUPPORT_ENABLED := 1
export OMP_NUM_THREADS := 1
export OMP_PLACES := threads
export OMP_PROC_BIND := spread

build:
	rm -rf build && \
	mkdir build && \
	cd build && \
	cmake -DSLATE_INSTALL=$$SLATE_INSTALL .. && \
	cmake --build . && \
	touch device_wrapper && \
	chmod +x device_wrapper && \
	echo "#!/bin/bash" > device_wrapper && \
	echo "export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID" >> device_wrapper && \
	echo "exec \$$*" >> device_wrapper && \
	echo "Build finished!" && \
	cd ..

alloc:
	@if [ -z "$$SLURM_JOB_ID" ]; then\
		salloc -N 4 -q interactive -t 03:00:00 -C gpu -G 16 -A m4341;\
	fi

small:
	srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 ./build/device_wrapper ./build/main data/small 11 8 2

australian:
	srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 ./build/device_wrapper ./build/main data/australian 690 14 2

svmguide1:
	srun -N 4 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 16 ./build/device_wrapper ./build/main data/svmguide1 3089 4 2

letter:
	srun -N 4 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 16 ./build/device_wrapper ./build/main data/letter 15000 5000 26
