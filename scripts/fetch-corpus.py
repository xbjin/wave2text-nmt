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
import zipfile


help_msg = ""

europarl_parallel = "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz"
europarl_mono = "http://www.statmt.org/wmt13/training-monolingual-europarl-v7.tgz"
news_parallel = "http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz"
news_mono = "http://www.statmt.org/wmt15/training-monolingual-nc-v10.tgz"
wmt14 = "http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/bitexts.tgz"
dev_v2 = "http://www.statmt.org/wmt15/dev-v2.tgz"
ted_fren = 'http://opus.lingfil.uu.se/download.php?f=TED2013/en-fr.txt.zip'
emea_fren = 'http://opus.lingfil.uu.se/download.php?f=EMEA/en-fr.txt.zip'

file_formats = {
    'europarl': ('europarl-v7.{src}-{trg}', europarl_parallel, 0),
    'europarl-mono': ('europarl-v7', europarl_mono, 0),
    'wmt14': (['ep7_pc45', 'nc9', 'ccb2_pc30', 'un2000_pc34', 'dev08_11', 'crawl'], wmt14, 0),
    'news-mono': ('news-commentary-v10', news_mono, 0),
    'news': ('news-commentary-v10.{src}-{trg}', news_parallel, 0),
    'news-test': (['newstest2011', 'newstest2012'], dev_v2, 0),
    'news-dev': ('newstest2013', dev_v2, 0),
    'TED' : ('TED2013.{trg}-{src}', ted_fren, 1),
    'EMEA': ('EMEA.{trg}-{src}',emea_fren, 1)
}


def concat_files_(args, unzip_folder_path):    
    langs = args.ext
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
  os.makedirs(new_path)  
  if tarfile.is_tarfile(gz_path) :
      with tarfile.open(gz_path, "r:gz") as corpus_tar:
          corpus_tar.extractall(new_path)          
  elif (os.path.splitext(gz_path)[1] == ".zip"):
      with zipfile.ZipFile(gz_path, "r") as z:
          z.extractall(new_path)      
  else:
      with gzip.open(gz_path, 'rb') as infile:
        filename = os.path.splitext(gz_path.split('/')[-1])[0]
        with open(os.path.join(new_path, filename), 'w') as outfile:
            for line in infile:
                outfile.write(line)


def gunzip_file(gz_path, new_path, args):
  # unzip main archive
  gzip_(gz_path, new_path)             
     
  # corpus is still an archive
  for root, directories, filenames in os.walk(new_path):
    for filename in filenames: 
      name ,ext = os.path.splitext(os.path.join(root,filename))
      if any(f in name for f in args.corp_filenames) and ext in [".tgz",".gz"]:
          gzip_(os.path.join(root,filename), new_path)


def maybe_download(exp_dir, args):
    unzip_folder = args.corpus
    url_corpus = file_formats[args.corpus][1]
    
    all_dir = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, f))] 
    unzip_folder_path = os.path.join(exp_dir, unzip_folder)
    if args.reset or not any(unzip_folder in s for s in all_dir):
        filename = url_corpus.split('/')[-1]
        filepath = os.path.join(exp_dir, filename)
        if not os.path.exists(filepath):
            filepath, _ = urllib.request.urlretrieve(url_corpus, filepath)
            statinfo = os.stat(filepath)
            print("Succesfully downloaded", filepath, statinfo.st_size, "bytes")      

        gunzip_file(filepath, unzip_folder_path, args) 
        args.path_to_archive = filepath
    return unzip_folder_path


def call_build_trilingual_corpus(args):
    # we sort files according to order given in extension input
    langs = args.ext
    args.corpus_lang.sort(key=lambda (x): langs.index(os.path.splitext(x)[-1][1:]))    
    
    print(args.corpus_lang)
    file_without_ext = [os.path.splitext(f)[0] for f in args.corpus_lang]
    basename_ = os.path.basename(os.path.splitext(file_without_ext[0])[0] + '.' + '-'.join(langs))

    # base
    args_ = ['scripts/build-trilingual-corpus.py']  \
            + file_without_ext \
            + [os.path.join(args.output_dir, basename_)] \
            + langs
    
    print("Calling build-trilingual-corpus with params : ", args_)
    subprocess.call(args_, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def fetch_corpus(args):
    exp_dir = args.tmp
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)    
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)  

    files = file_formats[args.corpus][0]    
    args.corp_filenames = [files] if type(files) is not list else files 
    
    unzip_folder_path = maybe_download(exp_dir, args)

    # getting all file unzipped with their path
    all_corpus_lang = []
    for root, directories, filenames in os.walk(unzip_folder_path):
        for filename in filenames: 
            _,ext = os.path.splitext(os.path.join(root,filename))
            if ext not in [".tgz",".gz"]:
                all_corpus_lang.append(os.path.join(root,filename))                

    corpus_wanted = []
    if args.corpus_type == 'mono':
        for ext in args.ext:
            corpus_wanted += [f + '.' + ext for f in args.corp_filenames]
    elif args.corpus_type == 'parallel':
        src_ext, trg_ext = args.ext[:-1], args.ext[-1]
        for ext in src_ext:
            corpus_wanted += [f.format(src=ext, trg=trg_ext) for f in args.corp_filenames]
    else:
        src_ext, trg_ext = args.ext[:-1], args.ext[-1]
        if len(src_ext) != 2:
            sys.exit("Trilingual mode requires exactly 2 source extensions")
        for ext in src_ext:
            corpus_wanted += [f.format(src=ext, trg=trg_ext) + '.' + ext for f in args.corp_filenames]

    print(corpus_wanted)
    # extracting corpus we want from all corpus
    args.corpus_lang = [c for c in all_corpus_lang if any(w in c for w in corpus_wanted)]

    if type(files) is list:
        args.corpus_lang = concat_files_(args, unzip_folder_path)
      
    # if trilingual given, call script
    if args.corpus_type == "trilingual":
        call_build_trilingual_corpus(args) 
        
    # copying file in output dir
    for f in args.corpus_lang:
        copyfile(f, os.path.join(args.output_dir, args.corpus + os.path.splitext(os.path.basename(f))[1]))
    
    shutil.rmtree(unzip_folder_path)
    #del archive ?
    if(file_formats[args.corpus][2]):
        os.remove(args.path_to_archive)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='corpus name')
    parser.add_argument('corpus_type', help='parallel or mono', choices=['parallel', 'mono', 'trilingual'])
    parser.add_argument('ext', nargs='+', help='list of extensions (target is last)')
    parser.add_argument('output_dir', help='destination directory')
    parser.add_argument('--tmp', help='directory where the files will be downloaded', default='tmp')
    parser.add_argument('--reset', help='overwrite previous files', action='store_true')

    args = parser.parse_args()
    
    fetch_corpus(args) 
    #fetch_testdev(args)
