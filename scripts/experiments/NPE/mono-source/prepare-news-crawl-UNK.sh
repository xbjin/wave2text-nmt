#!/usr/bin/env bash




#part1
	# appending unk to trg vocab and tokenizing corpuses

#part2
	# pretraining model 



cur_dir=`pwd`
script_dir=${cur_dir}/scripts  
data_dir_crawl=data/mono/news-crawl
data_dir_crawl_UNK=data/mono/news-crawl-UNK
corpus_crawl=news-crawl

data_dir_PE=data/simulated-PE/OMPI
corpus_PE=OMPI

src=fr
trg=en

######################
#       part1        #
######################


#take back the already merged voc
cp ${data_dir_crawl}/with-OMPI/vocab.en ${data_dir_crawl_UNK}/with-OMPI/vocab.en
cp ${data_dir_crawl}/with-OMPI/vocab.en ${data_dir_crawl_UNK}/with-OMPI/vocab.mt

#this call with append unk to merged voc and tokenize the corpuses with the unked voc and align
#ps : if voc size is 60K and we append unk tokens, ouput size is still 60K (last words are deleted)
${script_dir}/prepare-data.py ${data_dir_crawl}/${corpus_crawl}.train ${trg} ${data_dir_crawl_UNK}/with-OMPI  \
--mode ids  \
--test-corpus ${data_dir_crawl}/${corpus_crawl}.test  \
--dev-corpus  ${data_dir_crawl}/${corpus_crawl}.dev  \
--vocab-path  ${data_dir_crawl_UNK}/with-OMPI/vocab.en  \
--output-prefix ${corpus_crawl}  \
--unk-align  \
--verbose  \
--threads 16 \


#copying original corpuses
cp ${data_dir_crawl}/${corpus_crawl}.dev.en ${data_dir_crawl_UNK}/with-OMPI/${corpus_crawl}.dev.en
cp ${data_dir_crawl}/${corpus_crawl}.test.en ${data_dir_crawl_UNK}/with-OMPI/${corpus_crawl}.test.en

#creating symbolic links

ln -s ~/seq2seq/${data_dir_crawl}/with-OMPI/news-crawl.test.ids.en ~/seq2seq/${data_dir_crawl_UNK}/with-OMPI/news-crawl.test.ids.mt 
ln -s ~/seq2seq/${data_dir_crawl}/with-OMPI/news-crawl.dev.ids.en ~/seq2seq/${data_dir_crawl_UNK}/with-OMPI/news-crawl.dev.ids.mt 

ln -s ~/seq2seq/${data_dir_crawl_UNK}/with-OMPI/news-crawl.dev.en ~/seq2seq/${data_dir_crawl_UNK}/with-OMPI/news-crawl.dev.mt 
ln -s ~/seq2seq/${data_dir_crawl_UNK}/with-OMPI/news-crawl.test.en ~/seq2seq/${data_dir_crawl_UNK}/with-OMPI/news-crawl.test.mt  

######################
#       part2       #
######################



