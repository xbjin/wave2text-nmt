#!/usr/bin/env bash


#part1
	# training embeddings on news-mono

#part2
	# pretraining model 



cur_dir=`pwd`
script_dir=${cur_dir}/scripts  
data_dir=data/mono/news-crawl
corpus=news-crawl
src=fr
trg=en


######################
#       part1        #
######################


# fetch news crawl
./scripts/fetch-corpus.py ${corpus} mono ${src} ${trg} ${data_dir}


# pre-process
${script_dir}/prepare-data.py ${data_dir}/${corpus} ${src} ${trg} ${data_dir} --output-prefix ${corpus} --suffix train \
  --mode prepare \
  --verbose \
  --normalize-digits \
  --normalize-punk \
  --normalize-moses \
  --remove-duplicates \
  --min 1 --max 0 \
  --dev-size 2000 \
  --test-size 2000 \
  --thread 16

######################
#       part2        #
######################


${script_dir}/multivec-mono --train  ${data_dir}/${corpus}.train.${src} \
			    --dimension 1024 \
			    --iter 1 \
			    --verbose \
			    --save-vectors ${corpus}.vectors.en
		     	    --save ${corpus}.en.bin
