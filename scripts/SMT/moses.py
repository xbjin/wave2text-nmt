#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import os
import sys
import logging
import argparse

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

help_msg = """Train a translation model using Moses (full pipeline)."""

commands = """\
mkdir -p {output_dir}
$MOSES_DIR/bin/lmplz -o {lm_order} < {lm_corpus}.{trg_ext} > {output_dir}/{lm_corpus_name}.arpa.{trg_ext}
$MOSES_DIR/bin/build_binary {output_dir}/{lm_corpus_name}.arpa.{trg_ext} {output_dir}/{lm_corpus_name}.blm.{trg_ext}
$MOSES_DIR/scripts/training/train-model.perl -root-dir "{output_dir}" \
-corpus {corpus} -f {src_ext} -e {trg_ext} -alignment grow-diag-final-and \
-reordering msd-bidirectional-fe -lm 0:3:{output_dir}/{lm_corpus_name}.blm.{trg_ext}:8 \
-mgiza -external-bin-dir $GIZA_DIR \
-mgiza-cpus {threads} -cores {threads} --parallel
$MOSES_DIR/bin/processPhraseTableMin -in {output_dir}/model/phrase-table.gz -out {output_dir}/phrase-table -nscores 4 -threads {threads}
$MOSES_DIR/bin/processLexicalTableMin -in {output_dir}/model/reordering-table.wbe-msd-bidirectional-fe.gz -out {output_dir}/reordering-table -threads {threads}
cat {output_dir}/model/moses.ini | sed s@PhraseDictionaryMemory@PhraseDictionaryCompact@ | \
sed -r s@path=\(.*\)model/phrase-table.gz@path=\\\\1phrase-table@ | \
sed -r s@path=\(.*\)model/reordering-table.wbe-msd-bidirectional-fe.gz@path=\\\\1reordering-table@ > {output_dir}/moses.ini
$MOSES_DIR/scripts/training/mert-moses.pl {dev_corpus}.{src_ext} {dev_corpus}.{trg_ext} \
$MOSES_DIR/bin/moses {output_dir}/moses.ini --mertdir $MOSES_DIR/bin/ \
--no-filter-phrase-table --decoder-flags="-threads {threads}" &> {output_dir}/tuning.log \
--working-dir={output_dir}/mert-work
mv {output_dir}/mert-work/moses.ini {output_dir}/moses.ini.tuned
rm -rf {output_dir}/model {output_dir}/corpus {output_dir}/mert-work\
{output_dir}/giza.{src_ext}-{trg_ext} {output_dir}/giza.{trg_ext}-{src_ext}
"""

if __name__ == '__main__':
    if 'MOSES_DIR' not in os.environ or 'GIZA_DIR' not in os.environ:
        sys.exit('Environment variable not defined')

    parser = argparse.ArgumentParser(description=help_msg,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_dir')
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--src-ext', required=True)
    parser.add_argument('--trg-ext', required=True)
    parser.add_argument('--dev-corpus', required=True)
    parser.add_argument('--lm-corpus')
    parser.add_argument('--lm-order', type=int, default=3)
    parser.add_argument('--threads', type=int, default=16)

    args = parser.parse_args()

    args.lm_corpus = args.lm_corpus or args.corpus
    args.lm_corpus_name = os.path.basename(args.lm_corpus)

    args.output_dir = os.path.abspath(args.output_dir)
    args.corpus = os.path.abspath(args.corpus)
    args.lm_corpus = os.path.abspath(args.lm_corpus)
    args.dev_corpus = os.path.abspath(args.dev_corpus)

    commands = commands.strip().format(**vars(args))

    for cmd in commands.split('\n'):
        logging.info(cmd)
        ret_code = os.system(cmd)
        if ret_code != 0:
            sys.exit('Command failed')
