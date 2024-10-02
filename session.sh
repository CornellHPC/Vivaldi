#!/bin/bash
salloc --nodes 4 --qos interactive --time 00:10:00 --constraint gpu --gpus 4 --account m4341
