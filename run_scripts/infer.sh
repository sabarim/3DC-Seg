#!/bin/bash

PYTHONPATH=../../:$PYTHONPATH

cd ../

python -m torch.distributed.launch --nproc_per_node $1 main.py -c $2 --task infer --wts $3
