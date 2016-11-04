#!/usr/bin/env bash

raw_data_dir=data/raw
data_dir=experiments/IWSLT14/data

mkdir -p ${raw_data_dir} ${data_dir}
cur_dir=`pwd`
cd ${raw_data_dir}

# wget "http://wit3.fbk.eu/archive/2014-01/texts/de/en/de-en.tgz"
tar xzf de-en.tgz

for ext in de en; do
    cat de-en/train.tags.de-en.${ext} | grep -v '<url>' | grep -v '<talkid>' | grep -v '<keywords>' | \
    sed -e 's/<title>//g' | sed -e 's/<\/title>//g' | sed -e 's/<description>//g' | sed -e 's/<\/description>//g' \
    > IWSLT14.de-en.${ext}

    cat de-en/IWSLT14.TED*.${ext}.xml | grep '<seg id' $o | sed -e 's/<seg id="[0-9]*">\s*//g' | \
    sed -e 's/\s*<\/seg>\s*//g' | sed -e "s/\’/\'/g" > IWSLT14.test.de-en.${ext}
done

rm -rf de-en
cd  ${cur_dir}

mkdir -p ${data_dir}/tmp

for ext in de en; do
    cat ${raw_data_dir}/IWSLT14.de-en.${ext} | scripts/tokenizer.perl -threads 8 -l ${ext} | \
    scripts/lowercase.perl > ${data_dir}/tmp/IWSLT14.tok.${ext}
    cat ${raw_data_dir}/IWSLT14.test.de-en.${ext} | scripts/tokenizer.perl -threads 8 -l ${ext} | \
    scripts/lowercase.perl > ${data_dir}/test.${ext}
done
scripts/clean-corpus-n.perl -ratio 1.5 ${data_dir}/tmp/IWSLT14.tok de en ${data_dir}/tmp/IWSLT14.clean 1 50

# split corpus into train/dev
for ext in de en; do
    awk '{if (NR%23 == 0) print $0; }' ${data_dir}/tmp/IWSLT14.clean.${ext} > ${data_dir}/dev.${ext}
    awk '{if (NR%23 != 0) print $0; }' ${data_dir}/tmp/IWSLT14.clean.${ext} > ${data_dir}/train.${ext}
    head -n 2000 ${data_dir}/dev.${ext} > ${data_dir}/dev.2000.${ext}
done
rm -rf ${data_dir}/tmp

# extract vocabulary
scripts/prepare-data.py ${data_dir}/train de en ${data_dir} --mode vocab --min-count 3 --vocab-size 0
