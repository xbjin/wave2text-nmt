#!/usr/bin/env bash

set -e

if [[ -z  ${GPU} ]]
then
    echo "error: you need to set variable \$GPU"
    exit 1
fi

# binary input, word-level output

root_dir=models/btec_speech
data_dir=data/btec_speech
gpu_id=${GPU}
size=512
embedding_size="1024 512"
trg_vocab_size=`wc -l ${data_dir}/vocab.fr | cut -d' ' -f1`
num_samples=512
layers=2
dropout_rate=0.5
max_steps=0
steps_per_checkpoint=1000
steps_per_eval=2000
decay_factor=0.95
lstm=--use-lstm
bidir=--bidir
buckets="120 10 160 10 200 15 240 15 280 15 340 20 400 20"

echo "### training model"

mkdir -p ${root_dir}

parameters="--size ${size} --embedding-size ${embedding_size} --layers ${layers} \
--vocab-size ${trg_vocab_size} --ext feats fr ${lstm} ${bidir} \
--gpu-id ${gpu_id} --allow-growth ${buckets} --binary-input feats"
train_parameters="--train --verbose ${parameters} --dropout-rate ${dropout_rate} --beam-size 1 \
--max-steps ${max_steps} --learning-rate-decay-factor ${decay_factor} \
--steps-per-checkpoint ${steps_per_checkpoint} --steps-per-eval ${steps_per_eval}"

python -m translate ${data_dir} ${root_dir} ${train_parameters} --log-file ${root_dir}/log.txt
