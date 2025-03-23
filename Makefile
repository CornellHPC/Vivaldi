# Makefile

.PHONY: build alloc test small australian svmguide1 letter compare

export SLATE_INSTALL := $(shell cd ~/slate/_install && pwd) # Change slate directory as necessary
export mpi := cray
export blas := libsci
export CXX := CC
export CUDAHOSTCXX=/usr/bin/gcc-12
export DVS_MAXNODES := 1__
export SLATE_GPU_AWARE_MPI := 1
export MPICH_GPU_SUPPORT_ENABLED := 1
export OMP_NUM_THREADS := 1
export OMP_PLACES := threads
export OMP_PROC_BIND := spread

# CMake arguments
CMAKE_ARGS :=

# Check for convergence
ifeq ($(CONVERGENCE), 1)
    CMAKE_ARGS += -DCONVERGENCE=1
endif

# Run in basic mode (no convergence check or fine-grained timing)
ifeq ($(BASIC), 1)
    CMAKE_ARGS += -DBASIC=1
endif

# Library for SLATE linkage
CMAKE_ARGS += -DSLATE_INSTALL=$$SLATE_INSTALL

build:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	module load gcc-native/12.3 && \
	rm -rf build && \
	mkdir build && \
	cd build && \
	cmake $(CMAKE_ARGS) .. && \
	cmake --build . && \
	touch device_wrapper && \
	chmod +x device_wrapper && \
	echo '#!/bin/bash' >> device_wrapper && \
	echo 'export CUDA_VISIBLE_DEVICES=$$SLURM_LOCALID' >> device_wrapper && \
	echo 'exec $$*' >> device_wrapper && \
	echo "Build finished!" && \
	cd ..

ceecee:
	echo $(CMAKE_ARGS)

alloc:
	@if [ -z "$$SLURM_JOB_ID" ]; then\
		salloc -N 4 -q interactive -t 03:00:00 -C gpu -G 16 -A m4341;\
	fi

test:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	cd build && \
	salloc -N 1 -q interactive -t 00:01:00 -C gpu -G 4 -A m4341 \
	srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 \
	device_wrapper ctest --output-on-failure -O /tmp/output test_exe

debug:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	cd build && \
	salloc -N 1 -q interactive -t 01:00:00 -C gpu -G 4 -A m4341 \
	srun -N 1 --ntasks-per-node 1 --cpus-per-task 32 --cpu-bind cores -G 1 \
	device_wrapper cuda-gdb test_exe

small:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	salloc -N 1 -q interactive -t 00:01:00 -C gpu -G 4 -A m4341 \
	srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 \
	build/device_wrapper build/main -i data/small -m 11 -n 8 -k 2

australian:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	salloc -N 1 -q interactive -t 00:01:00 -C gpu -G 4 -A m4341 \
	srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 \
	build/device_wrapper build/main -i data/australian -m 690 -n 14 -k 2

svmguide1:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	salloc -N 4 -q interactive -t 00:01:00 -C gpu -G 16 -A m4341 \
	srun -N 4 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 16 \
	build/device_wrapper build/main -i data/svmguide1 -m 3089 -n 4 -k 2

letter:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	salloc -N 4 -q interactive -t 00:01:00 -C gpu -G 16 -A m4341 \
	srun -N 4 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 16 \
	build/device_wrapper build/main -i data/letter -m 15000 -n 5000 -k 26

rand:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	salloc -N 1 -q interactive -t 00:01:00 -C gpu -G 4 -A m4341 \
	srun -N 1 --ntasks-per-node 1 --cpus-per-task 32 --cpu-bind cores -G 1 \
	build/device_wrapper build/main -i data/rand -m 70000 -n 64 -k 128

profile:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	salloc -N 1 -q interactive -t 00:01:00 -C gpu -G 4 -A m4341 \
	srun -N 1 --ntasks-per-node 1 --cpus-per-task 32 --cpu-bind cores -G 1 \
	nsys profile --stats=true --cuda-memory-usage=true --trace=cuda,cublas,cusparse --output=/tmp/report \
	build/device_wrapper build/main -i data/rand -m 70000 -n 64 -k 128

compare:
	@if [ -z "$(file)" ]; then \
		echo "Error: file variable not set. Usage: make compare file=<filename>"; exit 1; \
	fi
	@if ! cmp -l data/$(file)_py data/$(file)_out > data/$(file)_diffs.txt; then \
		echo "Comparison failed: Differences found between data/$(file)_py and data/$(file)_out. See data/$(file)_diffs.txt for details."; \
	fi

