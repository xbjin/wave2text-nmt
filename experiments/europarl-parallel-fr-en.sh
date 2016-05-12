#!/bin/bash

#for fetch_corpus
corpus=europarl
corpus_type=parallel
corpus_test=news-test
corpus_dev=news-dev 

#for translation
data_dir=data/europarl-parallel-fr-en
train_dir=model/europarl-parallel-fr-en
size_layer=1024
num_layers=1
gpu_id=1


echo ">>>>>>>>> calling fetch_corpus"

./scripts/fetch-corpus.py ${corpus} ${corpus_type} fr ${data_dir}
./scripts/fetch-corpus.py ${corpus_test} mono fr en ${data_dir}
./scripts/fetch-corpus.py ${corpus_dev}  mono fr en ${data_dir}

echo ">>>>>>>>> calling prepare_data"

./scripts/prepare-data.py ${data_dir}/${corpus} fr en ${data_dir} --mode all \
                                                                  --verbose \
                                                                  --normalize-digits \
                                                                  --normalize-punk \
                                                                  --dev-corpus  ${data_dir}/${corpus_dev} \
                                                                  --test-corpus ${data_dir}/${corpus_test}

echo ">>>>>>>>> calling translation model"

python -m translate ${data_dir} ${train_dir} \
--size ${size_layer} \
--num-layers ${num_layers} \
--src-ext fr \
--trg-ext en \
--verbose \
--log-file ${train_dir}/log.txt \
--gpu-id ${gpu_id}
