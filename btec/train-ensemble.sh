#!/usr/bin/env bash

# set -e  # exit script on failure

# set as variable before running script
if [[ -z  ${GPU} ]]
then
    echo "error: you need to set variable \$GPU"
    exit 1
fi

export LD_LIBRARY_PATH="/usr/local/cuda/lib64/"

root_dir=btec/models/ensemble
data_dir=data/btec_fr-en
gpu_id=${GPU}
embedding_size=512
vocab_size=10000
num_samples=512
layers=1
dropout_rate=0.5
max_steps=30000
steps_per_checkpoint=1000
steps_per_eval=2000
decay_factor=0.95
lstm=
bidir=

mkdir -p ${root_dir}

parameters="--size ${embedding_size} --num-layers ${layers} --vocab-size ${vocab_size} \
--src-ext fr --trg-ext en ${lstm} ${bidir} --gpu-id ${gpu_id} --allow-growth"
train_parameters="--train --verbose ${parameters} --dropout-rate ${dropout_rate} --beam-size 1 \
--max-steps ${max_steps} --learning-rate-decay-factor ${decay_factor} \
--steps-per-checkpoint ${steps_per_checkpoint} --steps-per-eval ${steps_per_eval}"

echo "### training model 1"

train_dir=${root_dir}/model_1
python -m translate ${data_dir} ${train_dir} ${train_parameters} --log-file ${train_dir}/log.txt

echo "### training model 2"

train_dir=${root_dir}/model_2
python -m translate ${data_dir} ${train_dir} ${train_parameters} --log-file ${train_dir}/log.txt

echo "### training model 3"

train_dir=${root_dir}/model_3
python -m translate ${data_dir} ${train_dir} ${train_parameters} --log-file ${train_dir}/log.txt

echo "### training model 4"

train_dir=${root_dir}/model_4
python -m translate ${data_dir} ${train_dir} ${train_parameters} --log-file ${train_dir}/log.txt

echo "### evaluating ensemble"
python -m translate ${data_dir} ${root_dir} --reset --checkpoints ${root_dir}/model_{1,2,3,4}/checkpoints.fr_en/best \
--decode ${data_dir}/dev --output ${root_dir}/eval.beam_8.out --beam-size 8 --ensemble --verbose ${parameters} --no-gpu