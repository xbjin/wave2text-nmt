#!/usr/bin/env bash

python2 -m translate experiments/speech_translation/ensemble/finetune2.yaml --train -v
python2 -m translate experiments/speech_translation/ensemble/finetune3.yaml --train -v
python2 -m translate experiments/speech_translation/ensemble/finetune4.yaml --train -v
python2 -m translate experiments/speech_translation/ensemble/finetune1.yaml --train -v
python2 -m translate experiments/speech_translation/ensemble/finetune5.yaml --train -v