# Makefile

.PHONY: alloc build env small australian letter

export SLATE_INSTALL := $(shell cd ../slate/_install && pwd) # Change slate directory as necessary
export mpi := cray
export blas := libsci
export CXX := CC
export DVS_MAXNODES := 1__
export SLATE_GPU_AWARE_MPI := 0
export MPICH_GPU_SUPPORT_ENABLED := 1
export OMP_NUM_THREADS := 1
export OMP_PLACES := threads
export OMP_PROC_BIND := spread

alloc:
	salloc --nodes 4 --qos interactive --time 03:00:00 --constraint gpu --gpus 16 --account m4341

build:
	rm -rf build && \
	mkdir build && \
	cd build && \
	cmake -DSLATE_INSTALL=$$SLATE_INSTALL .. && \
	cmake --build . && \
	cd ..

small:
	srun --nodes=1 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=4 --gpu-bind=single:1 build/main data/small 11 8 4

australian:
	srun --nodes=1 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=4 --gpu-bind=single:1 build/main data/australian 690 14 2
# srun --nodes=4 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=16 --gpu-bind=single:1 build/main+pat data/australian 690 14 2

svmguide1:
	srun --nodes=4 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=16 --gpu-bind=single:1 build/main data/svmguide1 3089 4 2

letter:
	srun --nodes=4 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=16 --gpu-bind=single:1 build/main data/letter 15000 5000 26
