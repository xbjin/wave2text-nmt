#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from six.moves import urllib
from shutil import copyfile
import tarfile
import subprocess


help_msg = ""

europarl_parallel = "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz"
europarl_mono = "http://www.statmt.org/wmt13/training-monolingual-europarl-v7.tgz"
news_parallel = "http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz"
news_mono = "http://www.statmt.org/wmt15/training-monolingual-nc-v10.tgz"
bitext = "http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/bitexts.tgz"
dev_v2 = "http://www.statmt.org/wmt15/dev-v2.tgz"


file_formats = {
'europarl_trilingual': 'europarl-v7.{src}-{trg}.{src}',
'europarl_parallel': 'europarl-v7.{src}-{trg}',
'europarl_mono': 'europarl-v7.{src}',
'news_trilingual' : 'news-commentary-v10.{src}-{trg}.{src}',
'news_parallel' : 'news-commentary-v10.{src}-{trg}',
'news_mono' : 'news-commentary-v10.{src}',
'dev_v2' : 'newstest{year}.{trg}',
'dev_v2_concat' : 'newsdev1112.{trg}'
}



def concat_files(corpus_test, args, key):
    
    f1213 = [c for c in corpus_test if any(w in c for w in ['2011','2012'])]
    filenames = f1213
    with open(os.path.join(args.output_dir, file_formats[key+"_concat"].format(trg=args.trg_ext)), 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                
def gunzip_file(gz_path, new_path):
  print("Unpacking %s to %s" % (gz_path, new_path))  
  with tarfile.open(gz_path, "r") as corpus_tar:
      corpus_tar.extractall(new_path)
        

def maybe_download(exp_dir, folder, args):
    url_corpus = eval(folder)        
    filename = url_corpus.split('/')[-1]
    filepath = os.path.join(exp_dir, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(url_corpus, filepath)
        statinfo = os.stat(filepath)
        print("Succesfully downloaded", filepath, statinfo.st_size, "bytes")
    
    unzip_folder = os.path.join(exp_dir, folder)
    gunzip_file(filepath, unzip_folder)
    return unzip_folder
    
def call_build_trilingual_corpus(args):

    #we sort files according to order given in extension input
    args.corpus_lang.sort(key=lambda (x): args.src_ext.index(os.path.splitext(x)[-1][1:]))
    
    
    file_without_ext = [os.path.splitext(f)[0] for f in args.corpus_lang]
    src_ext_ = args.src_ext+[args.trg_ext] 
    basename_=os.path.basename(os.path.splitext(file_without_ext[0])[0]+'.'+'-'.join(src_ext_))

    #base
    args_ = ['scripts/build-trilingual-corpus.py']  \
            + file_without_ext \
            + [os.path.join(args.output_dir, basename_)] \
            + src_ext_

    
    print("Calling build-trilingual-corpus with params : ", args_)
    subprocess.call(args_, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  
def fetch_testdev(args):
    all_dir = [f for f in os.listdir(args.exp) if os.path.isdir(os.path.join(args.exp, f))]
    folder_unzip = "dev_v2"

    if not any(folder_unzip in s for s in all_dir):
        maybe_download(args.exp, folder_unzip, args) 
     
    all_test = []
    for root, directories, filenames in os.walk(os.path.join(args.exp, folder_unzip)):
        for filename in filenames: 
            all_test.append(os.path.join(root,filename))                
            
    key = folder_unzip
    corpus_wanted = [file_formats[key].format(year=y,trg=args.trg_ext) for y in [2011,2012,2013]]

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
        
        
    all_dir = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, f))]
    type_ = 'parallel' if(args.corpus_type == 'trilingual') else args.corpus_type
    folder_unzip = args.corpus+"_"+type_

    if not any(folder_unzip in s for s in all_dir):
        maybe_download(exp_dir, folder_unzip, args) 
    
    #getting all file unzipped with their path
    args.all_corpus_lang = []
    for root, directories, filenames in os.walk(os.path.join(exp_dir, folder_unzip)):
        for filename in filenames: 
            args.all_corpus_lang.append(os.path.join(root,filename))                
    
    key = args.corpus+"_"+args.corpus_type
    
    #getting coprus name we want
    corpus_wanted = [file_formats[key].format(src=l,trg=args.trg_ext) for l in args.src_ext]
    
    #extracting corpus we want from all corpus
    args.corpus_lang = [c for c in args.all_corpus_lang if any(w in c for w in corpus_wanted)]
    
    #if trilingual given, call script    
    if(args.corpus_type == "trilingual"):
        call_build_trilingual_corpus(args) 
        
    #copying file in output dir  
    for f in args.corpus_lang:
        copyfile(f, os.path.join(args.output_dir, os.path.basename(f)))        
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='corpus either europarl or news',
                        choices=['europarl', 'news'])
    parser.add_argument('corpus_type', help='parallel or mono',
                        choices=['parallel', 'mono', 'trilingual'])
    parser.add_argument('src_ext', nargs='+', help='list of ext for the corpus')   
    parser.add_argument('output_dir', help='output_dir') 
    parser.add_argument('--trg_ext', help='target ext', default='en') 
    parser.add_argument('--exp', help='path to expe directory', default='experiments')
   
    args = parser.parse_args()
    
    fetch_corpus(args) 
    fetch_testdev(args)

        