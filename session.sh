#!/bin/bash
salloc --nodes 4 --qos interactive --time 01:00:00 --constraint gpu --gpus 16 --account m4341
