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
ted = 'http://opus.lingfil.uu.se/download.php?f=TED2013/{src}-{trg}.txt.zip'
emea = 'http://opus.lingfil.uu.se/download.php?f=EMEA/{src}-{trg}.txt.zip'
news_crawl = 'http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.{ext}.shuffled.v2.gz'

file_formats = {
    'europarl': (['europarl-v7.{src}-{trg}.{ext}'], europarl_parallel),
    'europarl-mono': (['europarl-v7.{ext}'], europarl_mono),
    'WMT14': (['ep7_pc45.{ext}', 'nc9.{ext}', 'ccb2_pc30.{ext}', 'un2000_pc34.{ext}', 'dev08_11.{ext}', 'crawl.{ext}'], wmt14),
    'news-mono': (['news-commentary-v10.{ext}'], news_mono),
    'news': (['news-commentary-v10.{src}-{trg}.{ext}'], news_parallel),
    'news-test': (['newstest2011.{ext}', 'newstest2012.{ext}'], dev_v2),
    'news-dev': (['newstest2013.{ext}'], dev_v2),
    'TED' : (['TED2013.{src}-{trg}.{ext}'], ted),
    'EMEA': (['EMEA.{src}-{trg}.{ext}'],emea),
    'news-crawl': (['news.2014.{ext}.shuffled.v2'], news_crawl)
}


def extract(file_path, output_dir):
  print("Unpacking %s to %s" % (file_path, output_dir))
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  if tarfile.is_tarfile(file_path):   # tar.gz
      with tarfile.open(file_path, 'r:gz') as f:
          f.extractall(output_dir)
  elif zipfile.is_zipfile(file_path): # zip
      with zipfile.ZipFile(file_path, 'r') as f:
          f.extractall(output_dir)
  else:
      with gzip.open(file_path, 'rb') as infile:  # gz
        filename, _ = os.path.splitext(os.path.basename(file_path))
        with open(os.path.join(output_dir, filename), 'w') as outfile:
            for line in infile:
                outfile.write(line)


def maybe_download(urls, output_dir, unzip_dir):
    # urls: list of possible urls (return the first one that works)
    for url in urls:
        filename = url.split('/')[-1]  # FIXME
        path = os.path.join(output_dir, filename)  # some archives share filenames

        if os.path.exists(path):
            try:
                extract(path, unzip_dir)
                return output_dir
            except IOError:  # wrong format or bad archive
                try:
                    os.unlink(path)
                except IOError:
                    pass

        try:
            path, _ = urllib.request.urlretrieve(url, path)
            extract(path, unzip_dir)
            return output_dir
        except IOError:
            try:
                os.unlink(path)
            except IOError:
                pass
    return None


def flatten(directory):
    for file_or_dir in os.listdir(directory):
        file_or_dir = os.path.join(directory, file_or_dir)
        if os.path.isdir(file_or_dir):
            for file_ in os.listdir(file_or_dir):
                path = os.path.join(file_or_dir, file_)
                shutil.move(path, directory)
            os.rmdir(file_or_dir)  # dir should be empty by now


def concat_files(input_filenames, output_filename)
    pass


def fetch_corpus(args):
    filenames, url = file_formats[args.corpus]

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    unzip_dir = os.path.join(args.tmp, args.corpus)
    try:
        os.makedirs(unzip_dir)
    except OSError:
        pass

    if len(args.ext) == 1:
        ext, = args.ext
        urls = [(url.format(ext=ext),)]
    elif len(args.ext) == 2:
        src, trg = args.ext
        urls = set([
            (url.format(ext=ext, src=src, trg=trg), url.format(ext=ext, src=trg, trg=src))  # try both directions
            for ext in (src, trg)
        ])
    else:
        raise NotImplementedError('no trilingual corpus for now')

    for urls_ in urls:
        maybe_download(urls_, args.tmp, unzip_dir)

    flatten(unzip_dir)

    if len(args.ext) == 1:
        ext, = args.ext
        filenames = [filename.format(ext=ext) for filename in filenames]
        output_filename = os.path.join(args.output_dir, '{corpus}.{ext}'.format(corpus=args.corpus, ext=ext))
        concat_files(filenames, output_filename)

    elif len(args.ext) == 2:
        src, trg = args.ext
        for ext in args.ext:
            try:
                pass
            except:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus', help='corpus name')
    #parser.add_argument('corpus_type', help='parallel or mono', choices=['parallel', 'mono', 'trilingual'])
    parser.add_argument('ext', nargs='+', help='list of extensions (target is last)')
    parser.add_argument('output_dir', help='destination directory')
    parser.add_argument('--tmp', help='directory where the files will be downloaded', default='tmp')

    args = parser.parse_args()

    fetch_corpus(args)