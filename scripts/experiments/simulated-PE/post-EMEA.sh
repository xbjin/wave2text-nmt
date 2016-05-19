#!/usr/bin/env bash


#part1
	# This script concat all .out file numerically, giving our MT file
#part2
	# We now split MT and EMEA (reference), without tokenizing, to train and test 



data_dir=data/simulated-PE/EMEA
script_dir=scripts
root_dir=seq2seq
#root_dir=`pwd`

#part 1

cd ${root_dir}/${data_dir}/splits
`cat \$(ls -v *.out) > ../EMEA.mt`
cd ${root_dir}

#part 2

${script_dir}/prepare-data.py ${data_dir}/EMEA mt tok.en tok.fr  ${data_dir} --mode prepare   \
                                                              	     	     --verbose \
                                                              	     	     --no-tokenize \
                                                              	     	     --test-size 2000



