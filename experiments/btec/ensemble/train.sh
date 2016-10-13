#!/usr/bin/env bash

python2 -m translate experiments/btec/ensemble/model_1.yaml --train -v --gpu-id 0
python2 -m translate experiments/btec/ensemble/model_2.yaml --train -v --gpu-id 0
python2 -m translate experiments/btec/ensemble/model_3.yaml --train -v --gpu-id 0
python2 -m translate experiments/btec/ensemble/model_4.yaml --train -v --gpu-id 0
python2 -m translate experiments/btec/ensemble/model_5.yaml --train -v --gpu-id 0