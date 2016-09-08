#!/usr/bin/env bash

root_dir=experiments/btec/pre-training

# pre-train
python2 -m translate ${root_dir}/config/news.yaml --train -v --max-steps 4000
# train
python2 -m translate ${root_dir}/config/btec.yaml --train -v --max-steps 4000

# compare results with baseline
python2 -m translate ${root_dir}/config/btec.yaml --decode ${root_dir}/data/test1.fr --output ${root_dir}/results/test1.mt
python2 -m translate ${root_dir}/config/btec.yaml --decode ${root_dir}/data/test2.fr --output ${root_dir}/results/test2.mt

echo "Test 1"
scripts/score.py ${root_dir}/results/test1.mt ${root_dir}/data/test1.en
echo "Test 1 (multi-ref)"
scripts/score.py ${root_dir}/results/test1.mt ${root_dir}/data/test1.mref.en
echo "Test 2"
scripts/score.py ${root_dir}/results/test1.mt ${root_dir}/data/test2.en
echo "Test 2 (multi-ref)"
scripts/score.py ${root_dir}/results/test1.mt ${root_dir}/data/test2.mref.en