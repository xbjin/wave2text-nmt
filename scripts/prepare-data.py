#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import izip, islice
from random import shuffle
from contextlib import contextmanager
import argparse
import subprocess
import tempfile
import os
import logging
import sys
import shlex


help_msg = """\
Prepare a parallel corpus for Neural Machine Translation.

If a single corpus is specified, it will be split into train/dev/test corpora
according to the given train/dev/test sizes.

Additional pre-processing can be applied to these files, using external (Moses)
scripts, such as tokenization, punctuation normalization or lowercasing.
The corpus can be shuffled, and too long or too short sentences removed.

Usage example:
    scripts/prepare-data.py data/news fr en output --dev-corpus data/news-dev\
    --max 0 --lowercase --shuffle

This example will create 6 files in `output/`: train.fr, train.en, test.fr,\
 test.en, dev.fr and dev.en. These files will be tokenized and lowercased and\
 empty lines will be filtered out. `test` files will contain 6000 (default)\
 lines from input corpus `data/news`, and `train` will contain the remaining\
 lines. `dev` files will contain the (processed) lines read from\
 `data/news-dev`. These three output corpora will be shuffled.
"""

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"

_UNK = "UNK"
_START_VOCAB_BASIC = [_PAD, _GO, _EOS, _UNK]

__UNK = ["UNK"+str(i) for i in range(-7,8)+["n"]]
_START_VOCAB_UNK = [_PAD, _GO, _EOS]
_START_VOCAB_UNK.extend(__UNK)



PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
UNKS = range(3, 19) #14 + 0 + null

temporary_files = []


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


@contextmanager
def open_temp_files(num=1, mode='w', delete=False):
    files = []
    try:
        for _ in range(num):
            files.append(tempfile.NamedTemporaryFile(mode=mode, delete=delete))
            if not delete:
                temporary_files.append(files[-1].name)
        yield files
    finally:
        for file_ in files:
            file_.close()


def create_vocabulary(id_, args):
    if(id_ == len(args.extensions)-1 and id_ > 0 and args.align): #target language, align 
        start_vocab = _START_VOCAB_UNK
    else:
        start_vocab = _START_VOCAB_BASIC
        
    ext = args.extensions[id_]
    vocab_size = args.vocab_size[id_]
    filename = os.path.join(args.output_dir, 'train.{}'.format(ext))
    output_filename = os.path.join(args.output_dir, 'vocab{}.{}'.format(
        vocab_size, ext))

    logging.info('creating vocabulary {} from {}'.format(output_filename,
                                                         filename))
    vocab = {}
    with open(filename) as input_file,\
            open(output_filename, 'w') as output_file:
        for i, line in enumerate(input_file, 1):
            if i % 100000 == 0:
                logging.info(' processing line {}'.format(i))
            tokens = line.split()
            for w in tokens:
                vocab[w] = vocab.get(w, 0) + 1


        vocab_list = start_vocab + sorted(vocab, key=vocab.get, reverse=True)
        if 0 < vocab_size < len(vocab_list):
            vocab_list = vocab_list[:vocab_size]

        output_file.writelines(w + '\n' for w in vocab_list)

    return dict(map(reversed, enumerate(vocab_list)))



def create_ids(corpus, id_, vocab, args):
    if args.align:
        create_ids_with_align(corpus, id_, vocab, args)
        return

    filename = '{}.{}'.format(corpus, args.extensions[id_])
    output_filename = '{}.ids{}.{}'.format(corpus, args.vocab_size[id_],
                                           args.extensions[id_])

    with open(filename) as input_file,\
            open(output_filename, 'w') as output_file:
        for line in input_file:
            ids = [str(vocab.get(w, UNK_ID)) for w in line.split()]
            output_file.write(' '.join(ids) + '\n')


def process_file(corpus, id_, args):
    filename = '{}.{}'.format(corpus, args.extensions[id_])
    logging.info('processing ' + filename)

    lang = args.lang[id_]

    with open_temp_files(num=1) as output_, open(filename) as input_:
        output_ = output_[0]

        def path_to(script_name):
            if args.scripts is None:
                return script_name
            else:
                return os.path.join(args.scripts, script_name)

        processes = [['cat']]   # just copy file if there is no other operation
        if args.normalize_punk:
            processes.append([path_to('normalize-punctuation.perl'), '-l',
                              lang])
        if args.tokenize:
            processes.append([path_to('tokenizer.perl'), '-l', lang, '-threads',
                              str(args.threads)])
        if args.lowercase:
            processes.append([path_to('lowercase.perl')])
        if args.normalize_numbers:
            processes.append(['sed', 's/[[:digit:]]/0/g'])

        ps = None

        for i, process in enumerate(processes):
            stdout = output_ if i == len(processes) - 1 else subprocess.PIPE
            stdin = input_ if i == 0 else ps.stdout

            ps = subprocess.Popen(process, stdin=stdin, stdout=stdout,
                                  stderr=open('/dev/null', 'w'))

        ps.wait()

        return output_.name


def process_corpus(corpus, args, output_corpus=None):
    input_filenames = [process_file(corpus, i, args)
                 for i in range(len(args.extensions))]

    output_filenames = None
    if output_corpus is not None:
        output_filenames = ['{}.{}'.format(output_corpus, ext)
                            for ext in args.extensions]

    with open_files(input_filenames) as input_files,\
            (open_temp_files(len(input_filenames)) if not output_filenames
            else open_files(output_filenames, 'w')) as output_files:

        # (lazy) sequence of sentence tuples
        all_lines = (lines for lines in izip(*input_files) if
                     all(min_ <= len(line.split()) <= max_ for line, min_, max_
                         in zip(lines, args.min, args.max)))

        if args.shuffle:
            all_lines = list(all_lines)  # not lazy anymore
            shuffle(all_lines)


        for lines in all_lines:  # keeps it lazy if no shuffle
            for line, output_file in zip(lines, output_files):
                output_file.write(line)

        return [f.name for f in output_files]


def split_corpus(filenames, dest_corpora, extensions):
      
    with open_files(filenames) as input_files:
        for corpus, size in reversed(dest_corpora):  # puts train corpus last
            output_filenames = ['{}.{}'.format(corpus, ext)
                                for ext in extensions]
            with open_files(output_filenames, mode='w') as output_files:
                for input_file, output_file in zip(input_files, output_files):
                    output_file.writelines(islice(input_file, size))
                    # If size is None, this will read the whole file.
                    # That's why we put train last.



def create_align(lang_pair, output_dir, args):

    src, trg = lang_pair
    tmp_file = os.path.join(output_dir, 'temp_align')   
        
    with open(tmp_file,"w+") as ouput_file:
        with open(src) as textfile1, open(trg) as textfile2: 
            for x, y in izip(textfile1, textfile2):
                x = x.strip()
                y = y.strip()                
                ouput_file.write("{0} ||| {1}\n".format(x, y))                       
     
    
    ouput_align_path = os.path.join(output_dir, 'train.align') 
    ouput_align_file = open(ouput_align_path, 'w+')
    
    
    fast_align_location = os.path.join(args.scripts, args.fast_align_loc)
    _args = shlex.split(fast_align_location+" -i "+tmp_file+" -d -o -v -I "+ str(args.fast_align_iter))
    p = subprocess.Popen(_args, stdout=ouput_align_file, stderr=subprocess.PIPE)
    p.wait()
    ouput_align_file.flush()
    #result = p.stderr.read()
    
    os.remove(tmp_file)
    return ouput_align_file.name




def create_ids_with_align(corpus, id_, vocab, args):

    if(id_ == len(args.extensions)-1 and id_ > 0 and "train" in corpus): #target language, align for train
        print(corpus)
        align_file = '{}.{}'.format(corpus, "align")
        align_lines = [line.split() for line in open(align_file)]

        filename = '{}.{}'.format(corpus, args.extensions[id_])        
        output_filename = '{}.ids{}.{}'.format(corpus, args.vocab_size[id_],
                                               args.extensions[id_])
    
        with open(filename) as input_file,\
                open(output_filename, 'w') as output_file:
            for i,line in enumerate(input_file):
                ids = []
                align_pair = dict(item.split("-") for item in align_lines[i])
                align_pair = {int(v): int(k) for k, v in align_pair.items()}   #reverse for target:source 
                for j,w in enumerate(line.split()):
                    token = vocab.get(w, UNK_ID)
                    if(token==UNK_ID):       
                        pos_source = align_pair.get(j,j+8)#if align not found, make it so we chose UNKnull                        
                        offset = int(pos_source)-int(j)
                        if abs(offset)<7:           
                            token = UNKS[7+offset]
                        else:
                            token = UNKS[15] #UNKnull
                    ids.append(str(token))
                output_file.write(' '.join(ids) + '\n')
    else:
        create_ids(corpus, id_, vocab, args)
                

def create_lookup_dictionnary(filenames, output_dir, args):
 
    src_file, trg_file, align = filenames
    
    #creatings all alignments pair found - pair stored as key, occurence is value   
    dictionnary = {}
    with open(src_file) as f1,\
            open(trg_file) as f2,\
                open(align) as f3:
                   for x, y, z in izip(f1, f2, f3):
                       align_pair = dict(item.split("-") for item in z.strip().split())
                       xtokens = x.strip().split()
                       ytokens = y.strip().split()
                       for k, v in align_pair.items():
                           w=(xtokens[int(k)],ytokens[int(v)])#tuple
                           dictionnary[w] = dictionnary.get(w, 0) + 1
      

    #taking all pair that have more than SELECTION_VALUE occurence                        
    dict_top = dict( (key, value) for (key, value) in dictionnary.items() if value >= args.dict_val_select)
    
    #taking most occurence  for a same source word
    highest_prob_dict = {}       
    for w in sorted(dict_top, key=dict_top.get, reverse=True):
        #print(w, dict_top[w])
        highest_prob_dict.setdefault(w[0], w[1])

            
            
    ouput_dict = os.path.join(output_dir, 'lookup_dict') 
    
    #writing the final dict
    with open(ouput_dict, "w+") as output_file:
        for key, value in highest_prob_dict.iteritems():
            output_file.write(key + ' ' + value + '\n')

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='training corpus')
    parser.add_argument('extensions', nargs='+', help='list of extensions')

    parser.add_argument('output_dir',
                        help='directory where the files will be copied')

    parser.add_argument('--dev-corpus', help='development corpus')
    parser.add_argument('--test-corpus', help='test corpus')

    parser.add_argument('--scripts', help='path to script directory '
                        '(None if in $PATH)', default='scripts')

    parser.add_argument('--dev-size', type=int,
                        help='size of development corpus', default=3000)
    parser.add_argument('--test-size', type=int,
                        help='size of test corpus', default=6000)
    parser.add_argument('--train-size', type=int,
                        help='size of training corpus (defaults to maximum)')

    parser.add_argument('--lang', nargs='+', help='optional list of language '
                                                  'codes (when different '\
                                                  'than file extensions)')

    parser.add_argument('--normalize-punk', help='normalize punctuation',
                        action='store_true')
    parser.add_argument('--normalize-numbers', help='normalize numbers '
                        '(replace all digits with 0)', action='store_true')
    parser.add_argument('--lowercase', help='put everything to lowercase',
                        action='store_true')
    parser.add_argument('--shuffle', help='shuffle the corpus',
                        action='store_true')
    parser.add_argument('--no-tokenize', dest='tokenize',
                        help='no tokenization', action='store_false')

    parser.add_argument('--verbose', help='verbose mode', action='store_true')

    parser.add_argument('--min', nargs='+', type=int, default=1,
                        help='min number of tokens per line')
    parser.add_argument('--max', nargs='+', type=int, default=50,
                        help='max number of tokens per line (0 for no limit)')

    parser.add_argument('--vocab-size', nargs='+', type=int, help='size of '
                        'the vocabularies (0 for no limit, '
                        'default: no vocabulary).')
    parser.add_argument('--create-ids', help='create train, test and dev id '
                        'files. Vocab size needed', action='store_true')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--align', help='create alignments with fast align',
                        action='store_true')
    parser.add_argument('--dict_val_select', help='number of hits for a pair to get selected',
                        default=100)
    parser.add_argument('--fast_align_loc', help='Location of fast_align in scripts folder',
                        default="fast_align")
    parser.add_argument('--fast_align_iter', help='Numver of iteration in fast align learning',
                        default=5)

    args = parser.parse_args()

    def fixed_length_arg(name, value, length):
        if value is None:
            return value
        elif isinstance(value, int):
            return [value for _ in range(length)]
        elif len(value) == 1:
            return [value[0] for _ in range(length)]
        elif len(value) == length:
            return value
        else:
            sys.exit('wrong number of values for parameter {}'.format(name))

    n = len(args.extensions)
    args.min = fixed_length_arg('--min', args.min, n)
    args.max = fixed_length_arg('--max', args.max, n)
    args.vocab_size = fixed_length_arg('--vocab-size', args.vocab_size, n)

    args.max = [n if n > 0 else float('inf') for n in args.max]

    if args.lang is None:
        args.lang = args.extensions
    elif len(args.lang) != args.extensions:
        sys.exit('wrong number of values for parameter --lang')

    if args.verbose:
        logging.basicConfig(format='%(message)s', level=logging.INFO)

    if not os.path.exists(args.output_dir):
        logging.info('creating directory')
        os.makedirs(args.output_dir)

    output_train = os.path.join(args.output_dir, 'train')
    output_test = os.path.join(args.output_dir, 'test')
    output_dev = os.path.join(args.output_dir, 'dev')

    try:
        
        if args.dev_corpus:
            logging.info('processing dev corpus')
            process_corpus(args.dev_corpus, args, output_dev)
        if args.test_corpus:
            logging.info('processing test corpus')
            process_corpus(args.test_corpus, args, output_test)

        logging.info('processing train corpus')
        if args.dev_corpus and args.test_corpus:
            process_corpus(args.corpus, args, output_train)
        else:
            filenames = process_corpus(args.corpus, args)                
            dest_corpora = [(output_train, args.train_size)]
            if not args.test_corpus:
                dest_corpora.append((output_test, args.test_size))
            if not args.dev_corpus:
                dest_corpora.append((output_dev, args.dev_size))

            logging.info('splitting files')
            split_corpus(filenames, dest_corpora, args.extensions)
                
            if args.align:
                logging.info('creating alignement')  
                #alignements is only for a pair of language
                # 1st given is soure, last given is target
                lang_pair = ['{}.{}'.format(output_train, args.extensions[0]),
                             '{}.{}'.format(output_train, args.extensions[-1])]
                #align is needed for train only
                align_file = create_align(lang_pair, args.output_dir, args)
                
                #now make dictionnary according to alignement
                logging.info('creating lookup dictionnary')
                filenames = lang_pair + [align_file]
                create_lookup_dictionnary(filenames, args.output_dir, args)
                
            
        if args.vocab_size:           
            logging.info('creating vocab files')
            vocabs = [create_vocabulary(id_, args)
                      for id_ in range(len(args.extensions))]
            if args.create_ids:
                logging.info('creating ids')
                for corpus in [output_train, output_dev, output_test]:
                    for id_ in range(len(args.extensions)):
                        create_ids(corpus, id_, vocabs[id_], args)

    finally:
        logging.info('removing temporary files')
        for name in temporary_files:  # remove temporary files
            try:
                os.remove(name)
            except OSError:
                pass
