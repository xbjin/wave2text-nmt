#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import sys
from itertools import izip_longest
from contextlib import contextmanager
import os

help_msg = "merging works this way: \
taking first entry of voc1 \
taking first entry of voc2 \
taking second entry of voc1 \
taking second entry of voc2 \
... \
Until size args is reached"



  
  
@contextmanager
def open_files(names, mode='r'):
    files = []
    try:
        for name_ in names:
            files.append(open(name_, mode=mode))
        yield files
    finally:
        for file_ in files:
            file_.close()
            
            
def do_add(s, x):
  l = len(s)
  s.add(x)
  return len(s) != l
  
  
def process_line(line, args):
    if(line is not None):
        line = line.strip()
        added=do_add(args.ouput_voc_set,line)
        if(added):
            args.output_voc_list+=line                

            
def merge_voc(args):
    if len(args.vocs) > 2:
        sys.exit("max 2 voc")
        
    args.ouput_voc_set   = set() #for uniqueness
    args.output_voc_list = []
    with open_files([args.vocs[0], args.vocs[1]]) as files:
        for src_line, trg_line in izip_longest(*files):            
            process_line(src_line,args)
            if len(args.ouput_voc_set) == args.size:
                break            
            process_line(trg_line,args)
            if len(args.ouput_voc_set) == args.size:
                break        
  
    if not os.path.exists(args.output_dir):
          os.makedirs(args.output_dir)
    f = open(os.path.join(args.output_dir,'vocab-merged.'+args.lang), 'w+')  
    for item in args.output_voc_list:
      f.write("%s\n" % item)
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('vocs', nargs='+', help='list of vocs')
    parser.add_argument('lang', help='lang')
    parser.add_argument('output_dir', help='destination directory')
    parser.add_argument('--size', help='size of output voc', default='60000',type=int)

    args = parser.parse_args()
    
    merge_voc(args)
