# Makefile

.PHONY: build blasbuild alloc test debug australian svmguide1 letter compare letter_combblas

export SLATE_INSTALL := $(shell cd ~/slate/_install && pwd)
export COMBBLAS_INSTALL := $(shell cd ~/CombBLAS/_install && pwd)
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

ACCOUNT = mXXXX

# CMake arguments
CMAKE_ARGS :=

# Run in basic mode (no convergence check or fine-grained timing)
ifeq ($(BASIC), 1)
    CMAKE_ARGS += -DBASIC=1
endif

# Library for SLATE linkage
CMAKE_ARGS += -DSLATE_INSTALL=$$SLATE_INSTALL -DCOMBBLAS_INSTALL=$$COMBBLAS_INSTALL

build:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	rm -rf build && \
	mkdir build && \
	cd build && \
	cmake $(CMAKE_ARGS) ../src && \
	cmake --build . && \
	touch device_wrapper && \
	chmod +x device_wrapper && \
	echo '#!/bin/bash' >> device_wrapper && \
	echo 'export CUDA_VISIBLE_DEVICES=$$SLURM_LOCALID' >> device_wrapper && \
	echo 'exec $$*' >> device_wrapper && \
	echo "Build finished!" && \
	cd ..

blasbuild:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	module load gcc-native/12.3 && \
	rm -rf blasbuild && \
	mkdir blasbuild && \
	cd blasbuild && \
	cmake $(CMAKE_ARGS) ../src_combblas && \
	cmake --build . && \
	touch device_wrapper && \
	chmod +x device_wrapper && \
	echo '#!/bin/bash' >> device_wrapper && \
	echo 'export CUDA_VISIBLE_DEVICES=$$SLURM_LOCALID' >> device_wrapper && \
	echo 'exec $$*' >> device_wrapper && \
	echo "Build finished!" && \
	cd ..

alloc:
	@if [ -z "$$SLURM_JOB_ID" ]; then\
		salloc -N 4 -q interactive -t 01:00:00 -C gpu -G 16 -A $(ACCOUNT);\
	fi

test:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	cd build && \
	srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 \
	device_wrapper ctest --output-on-failure -O /tmp/output test_exe

debug:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	cd build && \
	salloc -N 1 -q interactive -t 01:00:00 -C gpu -G 4 -A $(ACCOUNT) \
	srun -N 1 --ntasks-per-node 1 --cpus-per-task 32 --cpu-bind cores -G 1 \
	device_wrapper cuda-gdb test_exe

australian:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	salloc -N 1 -q interactive -t 00:01:00 -C gpu -G 4 -A $(ACCOUNT) \
	srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 \
	build/device_wrapper build/main -i data/australian -m 690 -n 14 -k 2

svmguide1:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	salloc -N 4 -q interactive -t 00:01:00 -C gpu -G 16 -A $(ACCOUNT) \
	srun -N 4 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 16 \
	build/device_wrapper build/main -i data/svmguide1 -m 3089 -n 4 -k 2

letter:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	srun -N 1 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 4 \
	build/device_wrapper build/main -i data/letter -m 15000 -n 5000 -k 26 -a 15d

letter_combblas:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	salloc -N 4 -q interactive -t 00:02:00 -C gpu -G 16 -A $(ACCOUNT) \
	srun -N 4 --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G 16 \
	blasbuild/device_wrapper blasbuild/main data/letter 15000 5000 26

rand:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	salloc -N 1 -q interactive -t 00:01:00 -C gpu -G 4 -A $(ACCOUNT) \
	srun -N 1 --ntasks-per-node 1 --cpus-per-task 32 --cpu-bind cores -G 4 \
	build/device_wrapper build/main -i data/rand -m 70000 -n 64 -k 128

profile:
	source /opt/cray/pe/lmod/lmod/init/bash && \
	module load cudatoolkit/12.2 && \
	salloc -N 1 -q interactive -t 00:01:00 -C gpu -G 4 -A $(ACCOUNT) \
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
