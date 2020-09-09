#!/bin/bash

PYTHONPATH=../../:$PYTHONPATH

cd ../../

python main.py -c run_configs/<name>.yaml --num_workers 16 --task train --wts None
