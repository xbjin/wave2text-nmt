#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import os
import gzip
import sys
from six.moves import urllib
from shutil import copyfile
import tarfile
import subprocess
import shutil

help_msg = ""

europarl_trilingual = "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz"
europarl_parallel = "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz"
europarl_mono = "http://www.statmt.org/wmt13/training-monolingual-europarl-v7.tgz"
news_trilingual = "http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz"
news_parallel = "http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz"
news_mono = "http://www.statmt.org/wmt15/training-monolingual-nc-v10.tgz"
bitext = "http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/bitexts.tgz"
dev_v2 = "http://www.statmt.org/wmt15/dev-v2.tgz"

#use it if folder name of unzip is not same as args.corpus name
redirect_url = {    
    'nc9' : 'bitext',
    'ccb2_pc30' : 'bitext',
    'crawl' : 'bitext',
    'ep7_pc45' : 'bitext',
    'un2000_pc34' : 'bitext',
    'dev08_11' : 'bitext'
}


file_formats = {
'europarl_trilingual': ['europarl-v7.{src}-{trg}.{src}'],
'europarl_parallel': ['europarl-v7.{src}-{trg}'],
'europarl_mono': ['europarl-v7.{src}'],
'news_trilingual' : ['news-commentary-v10.{src}-{trg}.{src}'],
'news_parallel' : ['news-commentary-v10.{src}-{trg}'],
'news_mono' : ['news-commentary-v10.{src}'],
'dev_v2' : ['newstest{year}.{trg}'],
'dev_v2_concat' : ['newsdev1112.{trg}'], 
'nc9' : ['nc9.{src}', 'nc9.{trg}'],
'ccb2_pc30' : ['ccb2_pc30.{src}', 'ccb2_pc30.{trg}'],
'crawl' : ['crawl.{src}', 'crawl.{trg}'],
'ep7_pc45' : ['ep7_pc45.{src}', 'ep7_pc45.{trg}'],
'un2000_pc34' : ['un2000_pc34.{src}', 'un2000_pc34.{trg}'],
'dev08_11' : ['dev08_11.{src}', 'dev08_11.{trg}']       
}



def concat_files(corpus_test, args, key):
    
    f1213 = [c for c in corpus_test if any(w in c for w in ['2011','2012'])]
    filenames = f1213
    
    with open(os.path.join(args.output_dir, file_formats[key+"_concat"][0].format(trg=args.trg_ext)), 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


def gzip_(gz_path, new_path):
  print("Unpacking %s to %s" % (gz_path, new_path))  
  if tarfile.is_tarfile(gz_path) :
      with tarfile.open(gz_path, "r:gz") as corpus_tar:
          corpus_tar.extractall(new_path)
  else:
      with gzip.open(gz_path, 'rb') as infile:
        filename = os.path.splitext(gz_path.split('/')[-1])[0]
        with open(os.path.join(new_path, filename), 'w') as outfile:
            for line in infile:
                outfile.write(line)
              
def gunzip_file(gz_path, new_path, args):
  #unzip main archive  
  gzip_(gz_path, new_path)             
     
  #corpus is still an archive           
  for root, directories, filenames in os.walk(new_path):
    for filename in filenames: 
      name ,ext = os.path.splitext(os.path.join(root,filename))
      if(args.corpus in name and ext in [".tgz",".gz"]):
          gzip_(os.path.join(root,filename), new_path)


def maybe_download(exp_dir, unzip_folder, args):
    # maybe download download the archive if it doesnt exits and unzip
    # in unzip_folder. Return folder and the key of the corpus file_formats 
    try:
        url_corpus = eval(unzip_folder) 
        key = unzip_folder
    except NameError:
        key = args.corpus
        unzip_folder = redirect_url[key]        
        url_corpus = eval(unzip_folder)
    
    
    
    all_dir = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, f))] 
    unzip_folder_path = os.path.join(exp_dir, unzip_folder)
    if not any(unzip_folder in s for s in all_dir):
        filename = url_corpus.split('/')[-1]
        filepath = os.path.join(exp_dir, filename)
        if not os.path.exists(filepath):
            filepath, _ = urllib.request.urlretrieve(url_corpus, filepath)
            statinfo = os.stat(filepath)
            print("Succesfully downloaded", filepath, statinfo.st_size, "bytes")
        

        gunzip_file(filepath, unzip_folder_path, args)
        
    return unzip_folder_path, key
    
def call_build_trilingual_corpus(args):

    #we sort files according to order given in extension input
    langs = args.src_ext+[args.trg_ext] 
    args.corpus_lang.sort(key=lambda (x): langs.index(os.path.splitext(x)[-1][1:]))
    
    
    file_without_ext = [os.path.splitext(f)[0] for f in args.corpus_lang]
    basename_=os.path.basename(os.path.splitext(file_without_ext[0])[0]+'.'+'-'.join(langs))

    #base
    args_ = ['scripts/build-trilingual-corpus.py']  \
            + file_without_ext \
            + [os.path.join(args.output_dir, basename_)] \
            + langs

    
    print("Calling build-trilingual-corpus with params : ", args_)
    subprocess.call(args_, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  
def fetch_testdev(args):
    
    folder_unzip = args.test_file
    folder_unzip_path, key = maybe_download(args.exp, folder_unzip, args) 
    all_test = []
    for root, directories, filenames in os.walk(folder_unzip_path):
        for filename in filenames: 
            all_test.append(os.path.join(root,filename))                
            

    corpus_wanted = [f.format(year=y,trg=args.trg_ext) for y in [2011,2012,2013]  for f in file_formats[key]]

    corpus_test = [c for c in all_test if any(w in c for w in corpus_wanted)]
    
    concat_files(corpus_test, args, key)
    
    for f in corpus_test:
        copyfile(f, os.path.join(args.output_dir, os.path.basename(f))) 
        
            
def fetch_corpus(args):
    exp_dir = args.exp
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)    
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)  

    
    unzip_folder = args.corpus+"_"+args.corpus_type        
    unzip_folder_path, key = maybe_download(exp_dir, unzip_folder, args) 
    
    #getting all file unzipped with their path
    all_corpus_lang = []
    for root, directories, filenames in os.walk(unzip_folder_path):
        for filename in filenames: 
            _,ext = os.path.splitext(os.path.join(root,filename))
            if(ext not in [".tgz",".gz"]):
                all_corpus_lang.append(os.path.join(root,filename))                
    
   
    #getting coprus name we want    
    key = unzip_folder if key is None else key
    corpus_wanted = [f.format(src=l,trg=args.trg_ext) for l in args.src_ext for f in file_formats[key]]
    #extracting corpus we want from all corpus
    args.corpus_lang = [c for c in all_corpus_lang if any(w in c for w in corpus_wanted)]
    #if trilingual given, call script    
    if(args.corpus_type == "trilingual"):
        call_build_trilingual_corpus(args) 
        
    #copying file in output dir  
    for f in args.corpus_lang:
        copyfile(f, os.path.join(args.output_dir, os.path.basename(f)))        
    
    shutil.rmtree(unzip_folder_path)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='corpus name')
    parser.add_argument('corpus_type', help='parallel or mono',
                        choices=['parallel', 'mono', 'trilingual'])
    parser.add_argument('src_ext', nargs='+', help='list of ext for the corpus')   
    parser.add_argument('output_dir', help='output_dir') 
    parser.add_argument('--trg_ext', help='target ext', default='en') 
    parser.add_argument('--test-file', help='test files to use', default='dev_v2')
    parser.add_argument('--exp', help='path to expe directory', default='experiments')
   
    args = parser.parse_args()
    
    fetch_corpus(args) 
    fetch_testdev(args)

        