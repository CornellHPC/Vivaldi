#!/bin/bash
srun --ntasks 4 --gpus 16 build/main test 3
