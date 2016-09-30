#!/usr/bin/env bash

raw_data_dir=data/raw/btec.fr-en
raw_audio_dir=${raw_data_dir}/speech_fr
data_dir=experiments/speech/data   # output directory for the processed files (text and audio features)

mkdir -p ${raw_audio_dir} ${data_dir}

experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Agnes Agnes
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Fabienne Fabienne
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Helene Helene
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Loic Loic
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Marion Marion
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Michel Michel
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Philippe Philippe

experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Agnes Agnes
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Helene Helene
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Loic Loic

experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test1.fr ${raw_audio_dir}/test1-Agnes
experiments/speech/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test2.fr ${raw_audio_dir}/test2-Agnes

# 40 MFCC features + frame energy, with derivatives and second-order derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Agnes/* ${data_dir}/train.Agnes
scripts/extract-audio-features.py ${raw_audio_dir}/test1-Agnes/* ${data_dir}/test1.Agnes
scripts/extract-audio-features.py ${raw_audio_dir}/test2-Agnes/* ${data_dir}/test2.Agnes
scripts/extract-audio-features.py ${raw_audio_dir}/dev-Agnes/* ${data_dir}/dev.Agnes

# 40 MFCC features + frame energy without derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Agnes/* ${data_dir}/train.Agnes41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Fabienne/* ${data_dir}/train.Fabienne41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Helene/* ${data_dir}/train.Helene41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Loic/* ${data_dir}/train.Loic41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Marion/* ${data_dir}/train.Marion41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Michel/* ${data_dir}/train.Michel41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/train-Philippe/* ${data_dir}/train.Philippe41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/dev-Agnes/* ${data_dir}/dev.Agnes41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/dev-Helene/* ${data_dir}/dev.Helene41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/dev-Loic/* ${data_dir}/dev.Loic41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/test1-Agnes/* ${data_dir}/test1.Agnes41 --no-derivatives
scripts/extract-audio-features.py ${raw_audio_dir}/test2-Agnes/* ${data_dir}/test2.Agnes41 --no-derivatives

scripts/prepare-data.py ${raw_data_dir}/btec-dev-concat fr en ${data_dir} --max 0 --lowercase --output dev --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test1 --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test2 --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-train fr en ${data_dir} --max 0 --lowercase --output train

scripts/prepare-data.py ${raw_data_dir}/btec-dev-concat mref.en ${data_dir} --max 0 --lowercase --output dev --mode prepare --lang en
scripts/prepare-data.py ${raw_data_dir}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test1 --mode prepare --lang en
scripts/prepare-data.py ${raw_data_dir}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test2 --mode prepare --lang en

# Agnes is only used for development/testing, to test model's ability to generalize to other voices
scripts/audio-features-cat.py ${data_dir}/train.{Helene,Fabienne,Loic,Marion,Michel,Philippe}41 ${data_dir}/train-concat.feats41
cat ${data_dir}/train.{fr,fr,fr,fr,fr,fr} > ${data_dir}/train-concat.fr
cat ${data_dir}/train.{en,en,en,en,en,en} > ${data_dir}/train-concat.en


# samples for debugging
head -n 100 ${data_dir}/dev.en > ${data_dir}/dev.100.en
head -n 100 ${data_dir}/dev.fr > ${data_dir}/dev.100.fr
scripts/audio-features-head.py -n100 ${data_dir}/dev.Agnes ${data_dir}/dev.100.Agnes
scripts/audio-features-head.py -n100 ${data_dir}/dev.Agnes41 ${data_dir}/dev.100.Agnes41

head -n 1000 ${data_dir}/train.en > ${data_dir}/train.1000.en
head -n 1000 ${data_dir}/train.fr > ${data_dir}/train.1000.fr
scripts/audio-features-head.py -n1000 ${data_dir}/train.Agnes ${data_dir}/train.1000.Agnes
scripts/audio-features-head.py -n1000 ${data_dir}/train.Agnes41 ${data_dir}/train.1000.Agnes41