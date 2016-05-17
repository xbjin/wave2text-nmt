#!/usr/bin/env bash

data_dir=data/simulated-PE/TED
moses_config=model/europarl_SMT_fr-en/model/moses.ini

script_dir=scripts
root_dir=seq2seq

# translate (run this part locally)
ssh -nf brahms1 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 0 16"
ssh -nf brahms2 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 16 16"
ssh -nf brahms3 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 32 16"
ssh -nf bach1 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 48 16"
ssh -nf bach2 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 64 16"
ssh -nf bach3 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 80 16"
ssh -nf bach4 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 96 16"
ssh -nf bach4 "cd ${root_dir}; ${script_dir}/SMT/moses-parallel.py ${moses_config} ${data_dir}/splits 112 16"
