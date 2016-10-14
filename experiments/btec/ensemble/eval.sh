#!/bin/bash

config=experiments/btec/ensemble/model_1.yaml
ensemble_dir=experiments/btec/ensemble
model_dir=${ensemble_dir}/model_1
data_dir=experiments/btec/data

parameters="-v"
python2 -m translate ${config} --decode ${data_dir}/dev --output ${ensemble_dir}/dev.greedy.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test1 --output ${ensemble_dir}/test1.greedy.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test2 --output ${ensemble_dir}/test2.greedy.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/train.1000 --output ${ensemble_dir}/train.1000.greedy.out ${parameters}

parameters="-v --beam-size 8"
python2 -m translate ${config} --decode ${data_dir}/dev --output ${ensemble_dir}/dev.beam8.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test1 --output ${ensemble_dir}/test1.beam8.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test2 --output ${ensemble_dir}/test2.beam8.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/train.1000 --output ${ensemble_dir}/train.1000.beam8.out ${parameters}

parameters="-v --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2"
python2 -m translate ${config} --decode ${data_dir}/dev --output ${ensemble_dir}/dev.beam8.lm20.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test1 --output ${ensemble_dir}/test1.beam8.lm20.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test2 --output ${ensemble_dir}/test2.beam8.lm20.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/train.1000 --output ${ensemble_dir}/train.1000.beam8.lm20.out ${parameters}

parameters="-v --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2 --ensemble --checkpoints ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best"
python2 -m translate ${config} --decode ${data_dir}/dev --output ${ensemble_dir}/dev.beam8.lm20.ensemble.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test1 -v --output ${ensemble_dir}/test1.beam8.lm20.ensemble.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test2 -v --output ${ensemble_dir}/test2.beam8.lm20.ensemble.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/train.1000 -v --output ${ensemble_dir}/train.1000.beam8.lm20.ensemble.out ${parameters}
