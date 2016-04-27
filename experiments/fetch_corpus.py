# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:24:25 2016

@author: delbrouck
"""

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
   
    file_without_ext = [os.path.splitext(f)[0] for f in args.corpus_lang]
    shared_lang = os.path.splitext(args.corpus_lang[0])[0][-2:]
    extensions = args.extensions+[shared_lang]       
    
    args_ = [args.python, 'scripts/build-trilingual-corpus.py']    
    
    #insert files
    args_[len(args_):1] = file_without_ext

    #insert ouput (europarlv7.fr-es-en)
    basename_=os.path.basename(os.path.splitext(file_without_ext[0])[0]+'.'+'-'.join(extensions))
    args_[len(args_):1] = [os.path.join(args.output_dir, basename_)]

    #insert lang 
    args_[len(args_):1] = extensions
    
    print("Calling build-trilingual-corpus with params : ", args_)
    subprocess.call(args_, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    

def fetch_corpus(args):
    exp_dir = args.exp
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)    
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)   
        
        
    all_dir = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, f))]
    type_ = 'parallel' if(args.corpus_type == 'trilingual') else args.corpus_type
    folder = args.corpus+"_"+type_

    if not any(folder in s for s in all_dir):
        maybe_download(exp_dir, folder, args) 
    
    #getting all file unzipped
    args.all_corpus_lang = []
    for root, directories, filenames in os.walk(os.path.join(exp_dir, folder)):
        for filename in filenames: 
            args.all_corpus_lang.append(os.path.join(root,filename))                
   
   #getting file according to extension input
    args.corpus_lang = [c for c in args.all_corpus_lang if (os.path.splitext(c)[-1][1:] in args.extensions) ]
    if len(args.corpus_lang) < len(args.extensions):
        sys.exit("This corpus doesnt contain all of the ext(s) given")
    
    #we sort files according to order given in extension input
    args.corpus_lang.sort(key=lambda (x): args.extensions.index(os.path.splitext(x)[-1][1:]))


    #copying file in output dir  
    for f in args.corpus_lang:
        copyfile(f, os.path.join(args.output_dir, os.path.basename(f)))
    
    
    #if trilingual given, call script    
    if(args.corpus_type == "trilingual"):
        call_build_trilingual_corpus(args) 
        
        
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='corpus either europarl or news',
                        choices=['europarl', 'news'])
    parser.add_argument('corpus_type', help='parallel or mono',
                        choices=['parallel', 'mono', 'trilingual'])
    parser.add_argument('extensions', nargs='+', help='list of extensions for the corpus')   
    parser.add_argument('output_dir', help='output_dir') 
    parser.add_argument('--exp', help='path to expe directory', default='experiments')
    parser.add_argument('--python', help='python bin', default='python') 
   
    args = parser.parse_args()
    
    fetch_corpus(args) 
    

        