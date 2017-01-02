raw_audio_train=experiments/ted_speech/raw/train.wav
raw_audio_dev=experiments/ted_speech/raw/dev.wav

data_mfcc_train=experiments/ted_speech/data/train.feats41
data_mfcc_dev=experiments/ted_speech/data/dev.feats41

raw_txt_train_prefix=experiments/ted_speech/raw/train
raw_txt_dev_prefix=experiments/ted_speech/raw/dev

data_txt_dir=experiments/ted_speech/data

scripts/extract-audio-features.py $raw_audio_train/* --output=$data_mfcc_train
scripts/extract-audio-features.py $raw_audio_dev/* --output=$data_mfcc_dev

scripts/prepare-data.py $raw_txt_train_prefix en $data_txt_dir --max 0 --lowercase --output train --mode all
scripts/prepare-data.py $raw_txt_dev_prefix en $data_txt_dir --max 0 --lowercase --output dev --mode prepare
