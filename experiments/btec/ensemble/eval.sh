#!/bin/bash

config=experiments/speech_translation/ensemble/model_1.yaml
ensemble_dir=experiments/speech_translation/ensemble
model_dir=${ensemble_dir}/model_1
data_dir=experiments/speech/data

#python2 -m translate ${config} --decode ${data_dir}/dev.Agnes -v --output ${model_dir}/dev.Agnes.greedy.best
#python2 -m translate ${config} --decode ${data_dir}/dev.Michel -v --output ${model_dir}/dev.Michel.greedy.best
#python2 -m translate ${config} --decode ${data_dir}/dev.Marion -v --output ${model_dir}/dev.Marion.greedy.best
#python2 -m translate ${config} --decode ${data_dir}/test1.Agnes -v --output ${model_dir}/test1.Agnes.greedy.best
#python2 -m translate ${config} --decode ${data_dir}/test1.Michel -v --output ${model_dir}/test1.Michel.greedy.best
#python2 -m translate ${config} --decode ${data_dir}/test1.Marion -v --output ${model_dir}/test1.Marion.greedy.best
#python2 -m translate ${config} --decode ${data_dir}/train.1000.Agnes -v --output ${model_dir}/train.1000.Agnes.greedy.best
#python2 -m translate ${config} --decode ${data_dir}/train.1000.Michel -v --output ${model_dir}/train.1000.Michel.greedy.best
#python2 -m translate ${config} --decode ${data_dir}/train.1000.Marion -v --output ${model_dir}/train.1000.Marion.greedy.best

#python2 -m translate ${config} --decode ${data_dir}/dev.Agnes -v --output ${model_dir}/dev.Agnes.beam8.best --beam-size 8
#python2 -m translate ${config} --decode ${data_dir}/dev.Michel -v --output ${model_dir}/dev.Michel.beam8.best --beam-size 8
#python2 -m translate ${config} --decode ${data_dir}/dev.Marion -v --output ${model_dir}/dev.Marion.beam8.best --beam-size 8
#python2 -m translate ${config} --decode ${data_dir}/test1.Agnes -v --output ${model_dir}/test1.Agnes.beam8.best --beam-size 8
#python2 -m translate ${config} --decode ${data_dir}/test1.Michel -v --output ${model_dir}/test1.Michel.beam8.best --beam-size 8
#python2 -m translate ${config} --decode ${data_dir}/test1.Marion -v --output ${model_dir}/test1.Marion.beam8.best --beam-size 8
#python2 -m translate ${config} --decode ${data_dir}/train.1000.Agnes -v --output ${model_dir}/train.1000.Agnes.beam8.best --beam-size 8
#python2 -m translate ${config} --decode ${data_dir}/train.1000.Michel -v --output ${model_dir}/train.1000.Michel.beam8.best --beam-size 8
#python2 -m translate ${config} --decode ${data_dir}/train.1000.Marion -v --output ${model_dir}/train.1000.Marion.beam8.best --beam-size 8

#python2 -m translate ${config} --decode ${data_dir}/dev.Agnes -v --output ${model_dir}/dev.Agnes.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
#python2 -m translate ${config} --decode ${data_dir}/dev.Michel -v --output ${model_dir}/dev.Michel.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
#python2 -m translate ${config} --decode ${data_dir}/dev.Marion -v --output ${model_dir}/dev.Marion.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
#python2 -m translate ${config} --decode ${data_dir}/test1.Agnes -v --output ${model_dir}/test1.Agnes.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
#python2 -m translate ${config} --decode ${data_dir}/test1.Michel -v --output ${model_dir}/test1.Michel.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
#python2 -m translate ${config} --decode ${data_dir}/test1.Marion -v --output ${model_dir}/test1.Marion.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
#python2 -m translate ${config} --decode ${data_dir}/train.1000.Agnes -v --output ${model_dir}/train.1000.Agnes.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
#python2 -m translate ${config} --decode ${data_dir}/train.1000.Michel -v --output ${model_dir}/train.1000.Michel.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
#python2 -m translate ${config} --decode ${data_dir}/train.1000.Marion -v --output ${model_dir}/train.1000.Marion.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2

python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --ensemble --decode ${data_dir}/dev.Agnes -v --output ${ensemble_dir}/dev.Agnes.ensemble.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --ensemble --decode ${data_dir}/dev.Michel -v --output ${ensemble_dir}/dev.Michel.ensemble.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --ensemble --decode ${data_dir}/dev.Marion -v --output ${ensemble_dir}/dev.Marion.ensemble.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --ensemble --decode ${data_dir}/test1.Agnes -v --output ${ensemble_dir}/test1.Agnes.ensemble.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --ensemble --decode ${data_dir}/test1.Michel -v --output ${ensemble_dir}/test1.Michel.ensemble.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --ensemble --decode ${data_dir}/test1.Marion -v --output ${ensemble_dir}/test1.Marion.ensemble.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --ensemble --decode ${data_dir}/train.1000.Agnes -v --output ${ensemble_dir}/train.1000.Agnes.ensemble.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --ensemble --decode ${data_dir}/train.1000.Michel -v --output ${ensemble_dir}/train.1000.Michel.ensemble.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2
python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --ensemble --decode ${data_dir}/train.1000.Marion -v --output ${ensemble_dir}/train.1000.Marion.ensemble.beam8.lm20.best --beam-size 8 --lm-file ${data_dir}/btec.arpa --lm-weight 0.2

#python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,3,5}/checkpoints/best --decode ${data_dir}/dev.Agnes -v --output ${ensemble_dir}/dev.Agnes.ensemble3.beam8.best --beam-size 8 
#python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,3,5}/checkpoints/best --decode ${data_dir}/dev.Michel -v --output ${ensemble_dir}/dev.Michel.ensemble3.beam8.best --beam-size 8 
#python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,3,5}/checkpoints/best --decode ${data_dir}/dev.Marion -v --output ${ensemble_dir}/dev.Marion.ensemble3.beam8.best --beam-size 8 
#python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,3,5}/checkpoints/best --decode ${data_dir}/test1.Agnes -v --output ${ensemble_dir}/test1.Agnes.ensemble3.beam8.best --beam-size 8 
#python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,3,5}/checkpoints/best --decode ${data_dir}/test1.Michel -v --output ${ensemble_dir}/test1.Michel.ensemble3.beam8.best --beam-size 8 
#python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,3,5}/checkpoints/best --decode ${data_dir}/test1.Marion -v --output ${ensemble_dir}/test1.Marion.ensemble3.beam8.best --beam-size 8 
#python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,3,5}/checkpoints/best --decode ${data_dir}/train.1000.Agnes -v --output ${ensemble_dir}/train.1000.Agnes.ensemble3.beam8.best --beam-size 8 
#python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,3,5}/checkpoints/best --decode ${data_dir}/train.1000.Michel -v --output ${ensemble_dir}/train.1000.Michel.ensemble3.beam8.best --beam-size 8 
#python2 -m translate ${config} --checkpoints ${ensemble_dir}/model_{1,3,5}/checkpoints/best --decode ${data_dir}/train.1000.Marion -v --output ${ensemble_dir}/train.1000.Marion.ensemble3.beam8.best --beam-size 8 
