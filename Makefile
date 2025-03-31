build:
	cd build
	cmake --build .

interactive:
	#!/bin/bash
	salloc --nodes 4 --qos interactive --time 01:00:00 --constraint gpu --gpus 16 --account m4341

australian_py:
	cd tests && python test.py australian 16 2

australian_cpp:
	#!/bin/bash
	export DVS_MAXNODES=1__
	export OMP_NUM_THREADS=1
	export OMP_PLACES=threads
	export OMP_PROC_BIND=spread
	srun --nodes=4 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=16 --gpu-bind=single:1 build/main data/australian 690 14 2

australian_diffs:
	cmp -l results/australian_py results/australian_cpp > results/australian_diffs.txt

svmguide_py:
	cd tests && python test.py svmguide1 16 2

svmguide_cpp:
	#!/bin/bash
	export DVS_MAXNODES=1__
	export OMP_NUM_THREADS=1
	export OMP_PLACES=threads
	export OMP_PROC_BIND=spread
	srun --nodes=4 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=16 --gpu-bind=single:1 build/main data/svmguide1 3089 4 2

svmguide_diffs:
	cd results && cmp -l smvguide1_py smvguide1_cpp > svmguide1_diffs.txt

letter_py:
	cd tests && python test.py letter 16 26

letter_cpp:
	#!/bin/bash
	export DVS_MAXNODES=1__
	export OMP_NUM_THREADS=1
	export OMP_PLACES=threads
	export OMP_PROC_BIND=spread
	srun --nodes=4 --ntasks-per-node=4 --cpus-per-task=32 --cpu-bind=cores --gpus=16 --gpu-bind=single:1 build/main data/letter 15000 16 26

letter_diffs:
	cmp -l results/letter_py results/letter_cpp > results/letter_diffs.txt