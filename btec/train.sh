#!/usr/bin/env bash

set -e

if [[ -z  ${GPU} ]]
then
    echo "error: you need to set variable \$GPU"
    exit 1
fi

data_dir=data/btec_fr-en
root_dir=btec/models/large
gpu_id=${GPU}
embedding_size=1024
vocab_size=10000
num_samples=512
layers=3
dropout_rate=0.7
max_steps=0
steps_per_checkpoint=1000
steps_per_eval=2000
decay_factor=0.95
lstm=
bidir=--bidir

mkdir -p ${root_dir}

echo "### training model"

parameters="--train --verbose --src-ext fr --trg-ext en \
--size ${embedding_size} --num-layers ${layers} --vocab-size ${vocab_size} \
--dropout-rate ${dropout_rate} --beam-size 1 --max-steps ${max_steps} \
--learning-rate-decay-factor ${decay_factor} ${lstm} ${bidir} \
--steps-per-checkpoint ${steps_per_checkpoint} --steps-per-eval ${steps_per_eval} \
--gpu-id ${gpu_id} --allow-growth"

train_dir=${root_dir}/model_1
python -m translate ${data_dir} ${train_dir} ${parameters} --log-file ${train_dir}/log.txt