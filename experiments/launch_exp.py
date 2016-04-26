# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:24:25 2016

@author: delbrouck
"""

import argparse
import os
import sys
from six.moves import urllib
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
    filename = args.url_corpus.split('/')[-1]
    filepath = os.path.join(exp_dir, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(args.url_corpus, filepath)
        statinfo = os.stat(filepath)
        print("Succesfully downloaded", filepath, statinfo.st_size, "bytes")
    
    unzip_folder = os.path.join(exp_dir, folder)
    gunzip_file(filepath, unzip_folder)
    return unzip_folder
    

def fetch_corpus(args, unzipped=False):
    exp_dir = args.exp
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)    

    all_dir = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, f))]
    folder = args.corpus+"_"+args.corpus_type
    #si dossier (eg) europarl_mono existe pas, on download puis dézippe
    if not any(folder in s for s in all_dir):
        maybe_download(exp_dir, folder, args) 
    
    args.corpus_lang = []
    for root, directories, filenames in os.walk(os.path.join(exp_dir, folder)):
        for filename in filenames: 
            args.corpus_lang.append(os.path.join(root,filename))
                
   
    args.corpus_lang = [c for c in args.corpus_lang if any("."+ext in c for ext in args.extensions)]
    if len(args.corpus_lang) < len(args.extensions):
        sys.exit("This corpus doesnt contain all of the ext(s) given")
    
    
def call_prepare_data(args):
   
    #key is filename , value is list of extension with this filename
    #europarl-v7.fr-en : [en,fr]    
    corpus_dict = {}
    for i in args.corpus_lang:
        name,ext = os.path.splitext(i)
        ext = ext.replace(".","")
        if name not in corpus_dict:
            corpus_dict[name] = [ext]
        else:
            corpus_dict[name].append(ext)

    
    data_dir = os.path.join(args.output_dir,(args.corpus+'_'+args.corpus_type+'_')+'_'.join(args.extensions))
       
    for key, value in corpus_dict.iteritems():

        #base
        args_ = [args.python, 'scripts/prepare-data.py', key, data_dir, 
                  '--dev-size', '0', '--test-size', '0']
        
        #insert languages
        args_[3:1] = value
        
        #insert vocab size
        if(args.vocab_size):
            args_[len(args_):1] = ['--vocab-size', ' '.join(args.vocab_size)]
            
        #insert create ids
        if(args.create_ids):
            args_[len(args_):1] = ['--create-ids']
        
        #insert alignement
        if(args.align):
            args_[len(args_):1] = ['--align'] 
            if(args.dict_threshold): #default 100
                args_[len(args_):1] = ['--dict-threshold', args.dict_threshold] 
            if(args.fast_align_bin): #default fast_align
                args_[len(args_):1] = ['--fast-align-bin', args.fast_align_bin]             
            if(args.fast_align_iter): #default 5
                args_[len(args_):1] = ['--fast-align-iter', args.fast_align_iter]          
                        
                        
        print("Calling prepare_data.py with params : ", args_)
        subprocess.call(args_, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
 


    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='corpus either europarl or news',
                        choices=['europarl', 'news'])
    parser.add_argument('corpus_type', help='parallel or mono',
                        choices=['parallel', 'mono'])
    parser.add_argument('extensions', nargs='+', help='list of extensions for the corpus')  
    
    parser.add_argument('--exp', help='path to expe directory', default='experiments')
    parser.add_argument('--python', help='python bin', default='python') 
    
    #param for prepare_data.py
    parser.add_argument('--output-dir', help='directory where the files will be copied', 
                        default='data/')
    parser.add_argument('--vocab-size', nargs='+', type=str, help='size of '
                        'the vocabularies (0 for no limit, '
                        'default: no vocabulary)')    #TODO QUID QUAND PARALLEL AVEC EN ?
    parser.add_argument('--create-ids', help='create train, test and dev id '
                        'files', action='store_true') 
    parser.add_argument('--align', help='align target unknown words with the '
                        'source using special UNK IDs', action='store_true')
    parser.add_argument('--dict-threshold', help='min count of a word pair '
                        'in the dictionary', type=str) #if none, default 100
    parser.add_argument('--fast-align-bin', help='name of the fast_align '
                        'binary (relative to script directory)')  #if none, default fast_align
    parser.add_argument('--fast-align-iter', help='number of iterations in '
                        'fast_align', type=str) #if none, default 5

                      
    args = parser.parse_args()
    
    args.url_corpus = eval(args.corpus + "_" + args.corpus_type)

    fetch_corpus(args) 
    call_prepare_data(args)
                             