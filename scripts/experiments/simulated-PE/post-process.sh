#!/usr/bin/env bash

# part1
# This script concat all .out file numerically, giving our MT file
# part2
# We now split MT and EMEA (reference), without tokenizing, to train and test

if [ $# -ne 1 ]
then
    echo "Wrong number of parameters"
    exit 1
fi

corpus=$1

data_dir=data/simulated-PE/${corpus}
script_dir=scripts
root_dir=`pwd`

#part 1

cd ${root_dir}/${data_dir}/splits
cat $(ls -v *.out) > ../${corpus}.tok.mt
cd ${root_dir}

#part 2

${script_dir}/prepare-data.py ${data_dir}/${corpus}.tok mt en fr  ${data_dir} --mode prepare \
                                                                              --verbose \
                                                                              --no-tokenize \
                                                                              --dev-size 1000 \
                                                                              --test-size 2000 \
                                                                              --output-prefix ${corpus}
