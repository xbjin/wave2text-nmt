#!/usr/bin/env bash

srilm=bin/srilm/bin/i686-m64
data_dir=experiments/speech/data
order=3

mkdir -p ${data_dir}

# KenLM
# ${kenlm}/bin/lmplz -o ${order} < data/btec_fr-en/train.en > ${LM_dir}/btec.train.arpa.en
# SRILM
${srilm}/ngram-count -text ${data_dir}/train.en -lm ${data_dir}/btec.arpa -order ${order}

# weight=0.9   # weight of the BTEC LM
# BTEC (already processed)
## TED+news+europarl  (tokenized but not lowercased)
#scripts/lowercase.perl < data/raw/ted+news+euro.en | ${kenlm}/bin/lmplz -o ${order} \
#> ${LM_dir}/ted+news+euro.arpa.en
#
## filter big LM
#${kenlm}/bin/filter union ${LM_dir}/ted+news+euro.arpa.en ${LM_dir}/ted+news+euro.arpa.filtered.en \
#< data/btec_fr-en/train.en
#
## merge language models
#${srilm}/bin/i686-m64/ngram -lm ${LM_dir}/btec.train.arpa.en -mix-lm ${LM_dir}/ted+news+euro.arpa.filtered.en \
#-lambda ${weight} -write-lm ${LM_dir}/btec.train.mix.ted+news+euro.arpa.en -unk -order ${order}