#!/usr/bin/env bash

LM_dir=data/LM
order=5
kenlm=~/tools/kenlm
srilm=~/getalp-tools/SRILM_1_7_2_beta
weight=0.9   # weight of the BTEC LM

# BTEC (already processed)
${kenlm}/bin/lmplz -o ${order} < data/btec_fr-en/train.en > ${LM_dir}/btec.train.arpa.en

# TED+news+europarl  (tokenized but not lowercased)
scripts/lowercase.perl < data/raw/ted+news+euro.en | ${kenlm}/bin/lmplz -o ${order} \
> ${LM_dir}/ted+news+euro.arpa.en

# filter big LM
${kenlm}/bin/filter union ${LM_dir}/ted+news+euro.arpa.en ${LM_dir}/ted+news+euro.arpa.filtered.en \
< data/btec_fr-en/train.en

# merge language models
${srilm}/bin/i686-m64/ngram -lm ${LM_dir}/btec.train.arpa.en -mix-lm ${LM_dir}/ted+news+euro.arpa.filtered.en \
-lambda ${weight} -write-lm ${LM_dir}/btec.train.mix.ted+news+euro.arpa.en -unk -order ${order}