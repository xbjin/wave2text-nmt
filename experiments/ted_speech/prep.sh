raw_audio_train=experiments/ted_speech/raw/train.scp
raw_audio_dev=experiments/ted_speech/raw/dev.scp

data_mfcc_train=experiments/ted_speech/data/train
data_mfcc_dev=experiments/ted_speech/data/dev

raw_txt_train_prefix=experiments/ted_speech/raw/train
raw_txt_dev_prefix=experiments/ted_speech/raw/dev

data_txt_dir=experiments/ted_speech/data

mkdir -p $data_mfcc_train $data_mfcc_dev
while read line
do
   uttname=`basename $line .wav`
   feats=$data_mfcc_dev/${uttname}.feats41 	
   #scripts/extract-audio-features.py $raw_audio_train --output=$data_mfcc_train
   scripts/extract-audio-features.py $raw_audio_dev --output=$feats
done < $raw_audio_dev
scripts/extract-audio-features.py $raw_audio_dev --output=$data_mfcc_dev
#scripts/prepare-data.py $raw_txt_train_prefix en $data_txt_dir --max 0 --lowercase --output train --mode all
#scripts/prepare-data.py $raw_txt_dev_prefix en $data_txt_dir --max 0 --lowercase --output dev --mode prepare
