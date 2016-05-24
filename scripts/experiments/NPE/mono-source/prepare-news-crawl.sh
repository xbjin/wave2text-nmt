#!/usr/bin/env bash


#part1
	# fetch news crawl, tokenizing it
	
#part2
	# creating both news crawl and ompi vocab, then merging them

#part3
	# tokenizing corpuses

#part4
	# training embeddings on news-mono

#part5
	# pretraining model 



cur_dir=`pwd`
script_dir=${cur_dir}/scripts  
data_dir_crawl=data/mono/news-crawl
corpus_crawl=news-crawl

data_dir_PE=data/simulated-PE/OMPI
corpus_PE=OMPI

src=fr
trg=en


######################
#       part1        #
######################


# fetch news crawl
./scripts/fetch-corpus.py ${corpus_crawl} mono ${src} ${trg} ${data_dir_crawl}


# pre-process
${script_dir}/prepare-data.py ${data_dir_crawl}/${corpus_crawl} ${src} ${trg} ${data_dir_crawl} --output-prefix ${corpus_crawl} --suffix train \
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

${script_dir}/prepare-data.py ${data_dir_crawl}/${corpus_crawl}.train ${trg} ${data_dir_crawl} --mode vocab --vocab-size 60000

${script_dir}/prepare-data.py ${data_dir_PE}/${corpus_PE}.train ${trg} ${data_dir_PE} --mode vocab --vocab-size 60000

${script_dir}/merge-voc.py ${data_dir_crawl}/vocab.${trg} ${data_dir_PE}/vocab.${trg} ${trg} ${data_dir_crawl}/with-OMPI --size 60000



######################
#       part3       #
######################

${script_dir}/prepare-data.py ${data_dir_crawl}/${corpus_crawl}.train ${trg} ${data_dir_crawl}/with-OMPI \
--mode ids \
--test-corpus  ${data_dir_crawl}/${corpus_crawl}.test \ 
--dev-corpus ${data_dir_crawl}/${corpus_crawl}.dev \ 
--vocab-path ${data_dir_crawl}/with-OMPI/vocab-merged.en \ 
--output-prefix ${corpus_crawl} 


${script_dir}/prepare-data.py ${data_dir_PE}/${corpus_PE}.train ${trg} ${data_dir_crawl}/with-OMPI \
--mode ids  \
--test-corpus  ${data_dir_PE}/${corpus_PE}.test  \
--dev-corpus ${data_dir_PE}/${corpus_PE}.dev  \
--vocab-path ${data_dir_crawl}/with-OMPI/vocab-merged.en  \
--output-prefix ${corpus_PE} 

######################
#       part4        #
######################


${script_dir}/multivec-mono --train  ${data_dir_crawl}/${corpus_crawl}.train.${trg} \
			    --dimension 1024 \
			    --iter 1 \
			    --verbose \
			    --save-vectors ${corpus_crawl}.vectors.en
		     	    --save ${corpus_crawl}.en.bin


######################
#       part5       #
######################


mv ${data_dir_crawl}/with-OMPI/vocab-merged.en ${data_dir_crawl}/with-OMPI/vocab.en \

#we do en to en but model need different extension, lets create fake mt
ln -s ~/seq2seq/${data_dir_crawl}/with-OMPI/vocab.en ~/seq2seq/${data_dir_crawl}/with-OMPI/vocab.mt \
ln -s ~/seq2seq/${data_dir_crawl}/with-OMPI/vectors.en ~/seq2seq/${data_dir_crawl}/with-OMPI/vectors.mt \
ln -s ~/seq2seq/${data_dir_crawl}/with-OMPI/news-crawl.train.ids.en ~/seq2seq/${data_dir_crawl}/with-OMPI/news-crawl.train.ids.mt \
ln -s ~/seq2seq/${data_dir_crawl}/with-OMPI/news-crawl.test.ids.en ~/seq2seq/${data_dir_crawl}/with-OMPI/news-crawl.test.ids.mt \
ln -s ~/seq2seq/${data_dir_crawl}/with-OMPI/news-crawl.dev.ids.en ~/seq2seq/${data_dir_crawl}/with-OMPI/news-crawl.dev.ids.mt \

python2 -m ${data_dir_crawl}/with-OMPI/ model/NMT/mono/crawl-OMPI  \
--train-prefix news-crawl.train.ids \
--dev-prefix news-crawl.dev.ids \
--decode ${data_dir_crawl}/with-OMPI/news-crawl.test.ids  \
--eval ${data_dir_crawl}/with-OMPI/news-crawl.dev.ids  \
--size 1024  \
--gpu-id 1 \
--verbose \
--dropout-rate 0.2 \
--log-file model/NMT/mono/crawl-OMPI/log.txt \
--src-ext mt \
--trg-ext en \
--load-embeddings en en \
--steps-per-checkpoint 1000 \
--steps-per-eval 4000 \
--learning-rate-decay-factor 0.95 \
--src-vocab-size 60000 \
--trg-vocab-size 60000 \



