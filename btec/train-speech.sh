#!/usr/bin/env bash

set -e

if [[ -z  ${GPU} ]]
then
    echo "error: you need to set variable \$GPU"
    exit 1
fi

root_dir=models/btec_speech
data_dir=data/btec_speech
config=btec/speech.yaml
gpu_id=${GPU}

mkdir -p ${root_dir}

python -m translate ${data_dir} ${root_dir} --config ${config} --log-file ${root_dir}/log.txt --gpu-id ${GPU}
