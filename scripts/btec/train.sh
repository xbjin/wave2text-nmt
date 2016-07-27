#!/usr/bin/env bash

set -e

if [[ -z  ${GPU} ]]
then
    echo "error: you need to set variable \$GPU"
    exit 1
fi

root_dir=models/btec
data_dir=data/btec
gpu_id=${GPU}
embedding_size=256
src_vocab_size=`wc -l ${data_dir}/vocab.fr | cut -d' ' -f1`
trg_vocab_size=`wc -l ${data_dir}/vocab.en | cut -d' ' -f1`
num_samples=512
layers=2
dropout_rate=0.5
max_steps=100000
steps_per_checkpoint=1000
steps_per_eval=2000
decay_factor=0.95
lstm=--use-lstm
bidir=--bidir

echo "### training model"

mkdir -p ${root_dir}

parameters="--size ${embedding_size} --layers ${layers} --vocab-size ${src_vocab_size} ${trg_vocab_size} \
--ext fr en ${lstm} ${bidir} \
--gpu-id ${gpu_id} --allow-growth ${buckets}"
train_parameters="--train --verbose ${parameters} --dropout-rate ${dropout_rate} --beam-size 1 \
--max-steps ${max_steps} --learning-rate-decay-factor ${decay_factor} \
--steps-per-checkpoint ${steps_per_checkpoint} --steps-per-eval ${steps_per_eval}"

python -m translate ${data_dir} ${root_dir} ${train_parameters} --log-file ${root_dir}/log.txt
