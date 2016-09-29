#!/usr/bin/env bash

# raw btec data is assumed to be in `data/raw/btec.fr-en/btec-{train,dev-concat,test1,test2}.{fr,en,mref.en}`
# raw audio files are stored in `data/raw/btec.fr-en/speech_fr/{train,test1,test2,dev}

raw_data_dir=data/raw/btec.fr-en
raw_audio_dir=${raw_data_dir}/speech_fr
data_dir=experiments/speech/data   # output directory for the processed files (text and audio features)
voice=Agnes

mkdir -p ${raw_audio_dir} ${data_dir}

experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train ${voice}
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev ${voice}
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test1.fr ${raw_audio_dir}/test1 ${voice}
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test2.fr ${raw_audio_dir}/test2 ${voice}

# other voices
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Fabienne Fabienne
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Helene Helene
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Loic Loic
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Marion Marion
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Michel Michel
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Philippe Philippe
# experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Helene Helene
# experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Loic Loic

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

# other voices
scripts/extract-audio-features.py ${raw_audio_dir}/train-Fabienne/* ${data_dir}/train.Fabienne41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Helene/* ${data_dir}/train.Helene41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Loic/* ${data_dir}/train.Loic41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Marion/* ${data_dir}/train.Marion41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Michel/* ${data_dir}/train.Michel41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Philippe/* ${data_dir}/train.Philippe41 --no-derivatives
# scripts/extract-audio-features.py ${raw_audio_dir}/dev-Helene/* ${data_dir}/dev.Helene41 --no-derivatives
# scripts/extract-audio-features.py ${raw_audio_dir}/dev-Loic/* ${data_dir}/dev.Loic41 --no-derivatives

scripts/prepare-data.py ${raw_data_dir}/btec-dev-concat fr en ${data_dir} --max 0 --lowercase --output dev --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test1 --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test2 --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-train fr en ${data_dir} --max 0 --lowercase --output train

scripts/prepare-data.py ${raw_data_dir}/btec-dev-concat mref.en ${data_dir} --max 0 --lowercase --output dev --mode prepare --lang en
scripts/prepare-data.py ${raw_data_dir}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test1 --mode prepare --lang en
scripts/prepare-data.py ${raw_data_dir}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test2 --mode prepare --lang en

head -n 100 ${data_dir}/dev.en > ${data_dir}/dev.100.en
head -n 100 ${data_dir}/dev.fr > ${data_dir}/dev.100.fr
scripts/audio-features-head.py -n100 ${data_dir}/dev.feats ${data_dir}/dev.100.feats
scripts/audio-features-head.py -n100 ${data_dir}/dev.feats41 ${data_dir}/dev.100.feats41