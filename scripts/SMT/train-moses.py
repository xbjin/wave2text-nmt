#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import os
import sys
import logging
import argparse

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

help_msg = """Train a translation model using Moses.
Use `train-lm.py` to train a language model beforehand."""

commands = """\
$MOSES_DIR/scripts/training/train-model.perl -root-dir "{output_dir}" \
-corpus {corpus} -f {src_ext} -e {trg_ext} -alignment grow-diag-final-and \
-reordering msd-bidirectional-fe -lm 0:3:{lm_corpus}.blm.{trg_ext}:8 \
-mgiza -external-bin-dir $GIZA_DIR \
-mgiza-cpus {threads} -cores {threads} --parallel
mkdir {output_dir}/binarized
$MOSES_DIR/bin/processPhraseTableMin -in {output_dir}/model/phrase-table.gz -out {output_dir}/binarized/phrase-table -nscores 4 -threads {threads}
$MOSES_DIR/bin/processLexicalTableMin -in {output_dir}/model/reordering-table.gz -out {output_dir}/binarized/reordering-table -threads {threads}
cat {output_dir}/model/moses.ini | sed s@PhraseDictionaryMemory@PhraseDictionaryCompact@ | \
sed -r s@path=\(.*\)model/phrase-table.gz@path=\\\\1binarized/phrase-table@ | \
sed -r s@path=\(.*\)model/reordering-table.wbe-msd-bidirectional-fe.gz@path=\\\\1binarized/reordering-table@ > {output_dir}/binarized/moses.ini
"""

if __name__ == '__main__':
    if 'MOSES_DIR' not in os.environ or 'GIZA_DIR' not in os.environ:
        sys.exit('Environment variable not defined')

    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_dir')
    parser.add_argument('corpus')
    parser.add_argument('lm_corpus')
    parser.add_argument('src_ext')
    parser.add_argument('trg_ext')
    parser.add_argument('--threads', type=int, default=16)

    args = parser.parse_args()

    commands = commands.strip().format(**vars(args))

    for cmd in commands.split('\n'):
        logging.info(cmd)
        os.system(cmd)
