#!/usr/bin/env bash

python2 -m translate experiments/btec/multi_task/pre_training_mono_dummy.yaml --train -v   # pre-train for 50k steps
python2 -m translate experiments/btec/multi_task/finetuning_mono_dummy.yaml --train -v     # then finetune for 20k steps