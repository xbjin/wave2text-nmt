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

europarl_parallel = "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz"
europarl_mono = "http://www.statmt.org/wmt13/training-monolingual-europarl-v7.tgz"
news_parallel = "http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz"
news_mono = "http://www.statmt.org/wmt15/training-monolingual-nc-v10.tgz"
wmt14 = "http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/bitexts.tgz"
dev_v2 = "http://www.statmt.org/wmt15/dev-v2.tgz"


file_formats = {
    'europarl': ('europarl-v7.{src}-{trg}', europarl_parallel),
    'europarl-mono': ('europarl-v7', europarl_mono),
    'news-crawl-2007': '',
    'wmt14': (['ep7_pc45', 'nc9', 'ccb2_pc30', 'un2000_pc34', 'dev08_11', 'crawl'], wmt14),
    'news-mono': ('news-commentary-v10', news_mono),
    'news': ('news-commentary-v10.{src}-{trg}', news_parallel),
    'news-test': (['newstest2011', 'newstest2011'], dev_v2),
    'news-dev': ('newstest2013', dev_v2)
}


def concat_files_(args, unzip_folder_path):    
    langs = args.src_ext + [args.trg_ext] 
    outputs=[]
    for l in langs:
        output_file = os.path.join(unzip_folder_path, args.corpus+"-concat"+"."+l)
        filenames_lang = [c for c in args.corpus_lang if l in os.path.splitext(c)[1]]
        if len(filenames_lang) == 0:
            continue
        outputs.append(output_file)
        with open(output_file, 'w') as outfile:
            for fname in filenames_lang:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
    return outputs


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
      if(any(f in name for f in args.corp_filenames) and ext in [".tgz",".gz"]):
          gzip_(os.path.join(root,filename), new_path)


def maybe_download(exp_dir, args):
    unzip_folder = args.corpus
    url_corpus = file_formats[args.corpus][1]
    
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
    return unzip_folder_path
    
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
  
  
def fetch_corpus(args):
    exp_dir = args.exp
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)    
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)  

    files = file_formats[args.corpus][0]    
    args.corp_filenames = [files] if type(files) is not list else files 
    
    unzip_folder_path = maybe_download(exp_dir, args) 
    
    #getting all file unzipped with their path
    all_corpus_lang = []
    for root, directories, filenames in os.walk(unzip_folder_path):
        for filename in filenames: 
            _,ext = os.path.splitext(os.path.join(root,filename))
            if ext not in [".tgz",".gz"]:
                all_corpus_lang.append(os.path.join(root,filename))                

    corpus_wanted = []
    if args.corpus_type == 'mono':
        for ext in args.src_ext:
            corpus_wanted += [f + '.' + ext for f in args.corp_filenames]
    elif args.corpus_type == 'parallel':
        for ext in args.src_ext:
            corpus_wanted += [f.format(src=ext, trg=args.trg_ext) for f in args.corp_filenames]
    else:
        if len(args.src_ext) != 2:
            sys.exit("If trinlingual given, need exactly 2 src_ext")
        for ext in args.src_ext:
            corpus_wanted += [f.format(src=ext, trg=args.trg_ext) + '.' + ext for f in args.corp_filenames]

    print(corpus_wanted)
    #extracting corpus we want from all corpus
    args.corpus_lang = [c for c in all_corpus_lang if any(w in c for w in corpus_wanted)]
    

    if type(files) is list:
        args.corpus_lang = concat_files_(args, unzip_folder_path)
      
    #if trilingual given, call script    
    if args.corpus_type == "trilingual":
        call_build_trilingual_corpus(args) 
        
    #copying file in output dir  
    for f in args.corpus_lang:
        copyfile(f, os.path.join(args.output_dir, os.path.basename(f)))        
    
    #shutil.rmtree(unzip_folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='corpus name')
    parser.add_argument('corpus_type', help='parallel or mono', choices=['parallel', 'mono', 'trilingual'])
    parser.add_argument('src_ext', nargs='+', help='list of source extensions')
    parser.add_argument('output_dir', help='destination directory') 
    parser.add_argument('--trg-ext', help='target extension', default='en')
    parser.add_argument('--test-file', help='test files to use', default='news-test')
    parser.add_argument('--dev-file', help='test files to use', default='news-dev')
    parser.add_argument('--exp', help='directory where the files will be downloaded', default='experiments')
   
    args = parser.parse_args()
    
    calls = [[args.corpus, args.corpus_type], [args.test_file, 'mono'], [args.dev_file, 'mono']]
    for c in calls: 
        args.corpus = c[0]
        args.corpus_type = c[1]
        fetch_corpus(args) 
    #fetch_testdev(args)
