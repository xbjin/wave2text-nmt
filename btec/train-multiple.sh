#!/usr/bin/env bash

# set -e  # exit script on failure

# set as variable before running script
if [[ -z  ${GPU} ]]
then
    echo "error: you need to set variable \$GPU"
    exit 1
fi

export LD_LIBRARY_PATH="/usr/local/cuda/lib64/"

root_dir=btec/models/bidir
data_dir=data/btec_fr-en
gpu_id=${GPU}
vocab_size=10000
num_samples=512
max_steps=30000
steps_per_checkpoint=1000
steps_per_eval=2000
decay_factor=0.95

mkdir -p ${root_dir}

parameters="--train --verbose --vocab-size ${vocab_size} --beam-size 1 --max-steps ${max_steps} \
--learning-rate-decay-factor ${decay_factor} --steps-per-checkpoint ${steps_per_checkpoint} \
--steps-per-eval ${steps_per_eval} --gpu-id ${gpu_id} --allow-growth"
decode_parameters="--decode ${data_dir}/test --beam-size 8 --allow-growth -v --reset \
--vocab-size ${vocab_size} --gpu-id ${gpu_id}"

echo "### training model 1"

train_dir=${root_dir}/model_1
embedding_size=512
dropout_rate=0.5
layers=2
lstm=
bidir=--bidir

python -m translate ${data_dir} ${train_dir} ${parameters} --log-file ${train_dir}/log.txt \
--size ${embedding_size} --num-layers ${layers} --dropout-rate ${dropout_rate} ${lstm} ${bidir}

python2 -m translate ${data_dir} ${train_dir} ${decode_parameters} --output ${train_dir}/eval.beam_8.out \
--checkpoints ${train_dir}/checkpoints.fr_en/best --size ${embedding_size} --num-layers ${layers} ${lstm} ${bidir}

echo "### training model 2"

train_dir=${root_dir}/model_2
embedding_size=512
dropout_rate=0.0
layers=2
lstm=
bidir=--bidir

python -m translate ${data_dir} ${train_dir} ${parameters} --log-file ${train_dir}/log.txt \
--size ${embedding_size} --num-layers ${layers} --dropout-rate ${dropout_rate} ${lstm} ${bidir}

python2 -m translate ${data_dir} ${train_dir} ${decode_parameters} --output ${train_dir}/eval.beam_8.out \
--checkpoints ${train_dir}/checkpoints.fr_en/best --size ${embedding_size} --num-layers ${layers} ${lstm} ${bidir}

echo "### training model 3"

train_dir=${root_dir}/model_3
embedding_size=256
dropout_rate=0.5
layers=2
lstm=
bidir=--bidir

python -m translate ${data_dir} ${train_dir} ${parameters} --log-file ${train_dir}/log.txt \
--size ${embedding_size} --num-layers ${layers} --dropout-rate ${dropout_rate} ${lstm} ${bidir}

python2 -m translate ${data_dir} ${train_dir} ${decode_parameters} --output ${train_dir}/eval.beam_8.out \
--checkpoints ${train_dir}/checkpoints.fr_en/best --size ${embedding_size} --num-layers ${layers} ${lstm} ${bidir}

echo "### training model 4"

train_dir=${root_dir}/model_4
embedding_size=512
dropout_rate=0.5
layers=2
lstm=
bidir=

python -m translate ${data_dir} ${train_dir} ${parameters} --log-file ${train_dir}/log.txt \
--size ${embedding_size} --num-layers ${layers} --dropout-rate ${dropout_rate} ${lstm} ${bidir}

python2 -m translate ${data_dir} ${train_dir} ${decode_parameters} --output ${train_dir}/eval.beam_8.out \
--checkpoints ${train_dir}/checkpoints.fr_en/best --size ${embedding_size} --num-layers ${layers} ${lstm} ${bidir}

echo "### training model 5"

train_dir=${root_dir}/model_5
embedding_size=512
dropout_rate=0.5
layers=1
lstm=
bidir=

python -m translate ${data_dir} ${train_dir} ${parameters} --log-file ${train_dir}/log.txt \
--size ${embedding_size} --num-layers ${layers} --dropout-rate ${dropout_rate} ${lstm} ${bidir}

python2 -m translate ${data_dir} ${train_dir} ${decode_parameters} --output ${train_dir}/eval.beam_8.out \
--checkpoints ${train_dir}/checkpoints.fr_en/best --size ${embedding_size} --num-layers ${layers} ${lstm} ${bidir}

echo "### training model 6"

train_dir=${root_dir}/model_6
embedding_size=512
dropout_rate=0.5
layers=2
lstm=--use-lstm
bidir=

python -m translate ${data_dir} ${train_dir} ${parameters} --log-file ${train_dir}/log.txt \
--size ${embedding_size} --num-layers ${layers} --dropout-rate ${dropout_rate} ${lstm} ${bidir}

python2 -m translate ${data_dir} ${train_dir} ${decode_parameters} --output ${train_dir}/eval.beam_8.out \
--checkpoints ${train_dir}/checkpoints.fr_en/best --size ${embedding_size} --num-layers ${layers} ${lstm} ${bidir}