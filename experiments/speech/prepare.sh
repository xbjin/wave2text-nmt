#!/usr/bin/env bash

# raw btec data is assumed to be in `data/raw/btec.fr-en/btec-{train,dev-concat,test1,test2}.{fr,en,mref.en}`
# raw audio files are stored in `data/raw/btec.fr-en/speech_fr/{train,test1,test2,dev}

raw_data_dir=data/raw/btec.fr-en
raw_audio_dir=${raw_data_dir}/speech_fr
data_dir=experiments/speech/data   # output directory for the processed files (text and audio features)

mkdir -p ${raw_audio_dir} ${data_dir}

experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test1.fr ${raw_audio_dir}/test1
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test2.fr ${raw_audio_dir}/test2

# 40 MFCC features + frame energy, with derivatives and second-order derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train/* ${data_dir}/train.feats
scripts/extract-audio-features.py ${raw_audio_dir}/test1/* ${data_dir}/test1.feats
scripts/extract-audio-features.py ${raw_audio_dir}/test2/* ${data_dir}/test2.feats
scripts/extract-audio-features.py ${raw_audio_dir}/dev/* ${data_dir}/dev.feats

# 40 MFCC features + frame energy without derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train/* ${data_dir}/train.feats41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/test1/* ${data_dir}/test1.feats41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/test2/* ${data_dir}/test2.feats41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/dev/* ${data_dir}/dev.feats41 --no-derivatives

scripts/prepare-data.py ${raw_data_dir}/btec-dev-concat fr en ${data_dir} --max 0 --lowercase --output dev --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test1 --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test2 --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-train fr en ${data_dir} --max 0 --lowercase --output train

scripts/prepare-data.py ${raw_data_dir}/btec-dev-concat mref.en ${data_dir} --max 0 --lowercase --output dev --mode prepare --lang en
scripts/prepare-data.py ${raw_data_dir}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test1 --mode prepare --lang en
scripts/prepare-data.py ${raw_data_dir}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test2 --mode prepare --lang en