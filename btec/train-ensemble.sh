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
embedding_size=256
vocab_size=10000
num_samples=512
layers=2
dropout_rate=0.5
max_steps=50000
steps_per_checkpoint=1000
steps_per_eval=2000
decay_factor=0.95
lstm=--use-lstm
bidir=--bidir

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

echo "### training model 5"

train_dir=${root_dir}/model_5
python -m translate ${data_dir} ${train_dir} ${train_parameters} --log-file ${train_dir}/log.txt

echo "### evaluating ensemble - no LM"
python -m translate ${data_dir} ${root_dir} --reset --checkpoints ${root_dir}/model_{1,2,3,4,5}/checkpoints.fr_en/best \
--decode ${data_dir}/dev --output ${root_dir}/eval.beam_12.out --beam-size 12 --ensemble \
--verbose ${parameters} --gpu-id ${gpu_id} --log-file ${root_dir}/log.beam_12.out

echo "### evaluating ensemble - LM 40%"
python -m translate ${data_dir} ${root_dir} --reset --checkpoints ${root_dir}/model_{1,2,3,4,5}/checkpoints.fr_en/best \
--decode ${data_dir}/dev --output ${root_dir}/eval.beam_12.lm_40.out --beam-size 12 --ensemble --use-lm \
--model-weights 0.12 0.12 0.12 0.12 0.12 0.4 --verbose ${parameters} \
--gpu-id ${gpu_id} --log-file ${root_dir}/log.beam_12.lm_40.out

echo "### evaluating ensemble - big LM 20%"
python -m translate ${data_dir} ${root_dir} --reset --checkpoints ${root_dir}/model_{1,2,3,4,5}/checkpoints.fr_en/best \
--decode ${data_dir}/dev --output ${root_dir}/eval.beam_12.big_lm_20.out --beam-size 12 --ensemble --use-lm \
--model-weights 0.16 0.16 0.16 0.16 0.16 0.2 --verbose ${parameters} --lm-prefix big_lm --lm-order 5 \
--gpu-id ${gpu_id} --log-file ${root_dir}/log.beam_12.big_lm_20.out
