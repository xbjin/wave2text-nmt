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

ensemble_params="${ensemble_dir}/model_1/checkpoints/best ${ensemble_dir}/model_2/checkpoints/best ${ensemble_dir}/model_3/checkpoints/best ${ensemble_dir}/model_4/checkpoints/best ${ensemble_dir}/model_5/checkpoints/best"
parameters="-v --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2 --ensemble --checkpoints ${ensemble_params}"
python2 -m translate ${config} --decode ${data_dir}/dev --output ${ensemble_dir}/dev.beam8.lm20.ensemble.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test1 -v --output ${ensemble_dir}/test1.beam8.lm20.ensemble.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test2 -v --output ${ensemble_dir}/test2.beam8.lm20.ensemble.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/train.1000 -v --output ${ensemble_dir}/train.1000.beam8.lm20.ensemble.out ${parameters}

scripts/score.py experiments/btec/ensemble/dev.greedy.out experiments/btec/data/dev.en
scripts/score.py experiments/btec/ensemble/test1.greedy.out experiments/btec/data/test1.en
scripts/score.py experiments/btec/ensemble/test2.greedy.out experiments/btec/data/test2.en
scripts/score.py experiments/btec/ensemble/dev.greedy.out experiments/btec/data/dev.mref.en
scripts/score.py experiments/btec/ensemble/test1.greedy.out experiments/btec/data/test1.mref.en
scripts/score.py experiments/btec/ensemble/test2.greedy.out experiments/btec/data/test2.mref.en
scripts/score.py experiments/btec/ensemble/train.1000.greedy.out experiments/btec/data/train.1000.en

scripts/score.py experiments/btec/ensemble/dev.beam8.out experiments/btec/data/dev.en
scripts/score.py experiments/btec/ensemble/test1.beam8.out experiments/btec/data/test1.en
scripts/score.py experiments/btec/ensemble/test2.beam8.out experiments/btec/data/test2.en
scripts/score.py experiments/btec/ensemble/dev.beam8.out experiments/btec/data/dev.mref.en
scripts/score.py experiments/btec/ensemble/test1.beam8.out experiments/btec/data/test1.mref.en
scripts/score.py experiments/btec/ensemble/test2.beam8.out experiments/btec/data/test2.mref.en
scripts/score.py experiments/btec/ensemble/train.1000.beam8.out experiments/btec/data/train.1000.en

scripts/score.py experiments/btec/ensemble/dev.beam8.lm20.out experiments/btec/data/dev.en
scripts/score.py experiments/btec/ensemble/test1.beam8.lm20.out experiments/btec/data/test1.en
scripts/score.py experiments/btec/ensemble/test2.beam8.lm20.out experiments/btec/data/test2.en
scripts/score.py experiments/btec/ensemble/dev.beam8.lm20.out experiments/btec/data/dev.mref.en
scripts/score.py experiments/btec/ensemble/test1.beam8.lm20.out experiments/btec/data/test1.mref.en
scripts/score.py experiments/btec/ensemble/test2.beam8.lm20.out experiments/btec/data/test2.mref.en
scripts/score.py experiments/btec/ensemble/train.1000.beam8.lm20.out experiments/btec/data/train.1000.en

scripts/score.py experiments/btec/ensemble/dev.beam8.lm20.ensemble.out experiments/btec/data/dev.en
scripts/score.py experiments/btec/ensemble/test1.beam8.lm20.ensemble.out experiments/btec/data/test1.en
scripts/score.py experiments/btec/ensemble/test2.beam8.lm20.ensemble.out experiments/btec/data/test2.en
scripts/score.py experiments/btec/ensemble/dev.beam8.lm20.ensemble.out experiments/btec/data/dev.mref.en
scripts/score.py experiments/btec/ensemble/test1.beam8.lm20.ensemble.out experiments/btec/data/test1.mref.en
scripts/score.py experiments/btec/ensemble/test2.beam8.lm20.ensemble.out experiments/btec/data/test2.mref.en
scripts/score.py experiments/btec/ensemble/train.1000.beam8.lm20.ensemble.out experiments/btec/data/train.1000.en

scripts/score.py experiments/SMT/model/dev.mt experiments/btec/data/dev.en
scripts/score.py experiments/SMT/model/test1.mt experiments/btec/data/test1.en
scripts/score.py experiments/SMT/model/test2.mt experiments/btec/data/test2.en
scripts/score.py experiments/SMT/model/dev.mt experiments/btec/data/dev.mref.en
scripts/score.py experiments/SMT/model/test1.mt experiments/btec/data/test1.mref.en
scripts/score.py experiments/SMT/model/test2.mt experiments/btec/data/test2.mref.en
scripts/score.py experiments/SMT/model/train.1000.mt experiments/btec/data/train.1000.en