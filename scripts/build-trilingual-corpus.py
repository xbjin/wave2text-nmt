#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from itertools import izip
from collections import defaultdict
import argparse

help_msg = """\
Build a trilingual corpus from two bilingual corpora.

Usage example:
    build-trilingual-corpus.py news.de-en news.fr-en news.de-fr-en de fr en\
"""



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('corpus1', help='name of the first corpus (e.g. news.de-en)')
    parser.add_argument('corpus2', help='name of the second corpus (e.g. news.fr-en)')
    parser.add_argument('output', help='output corpus (e.g. news.de-fr-en)')

    parser.add_argument('lang1', help='non-shared lang of the first corpus (e.g. de)')
    parser.add_argument('lang2', help='non-shared lang of the second corpus (e.g. fr)')
    parser.add_argument('shared_lang', help='shared lang (e.g. en)')

    args = parser.parse_args()

    corpus1, corpus2, output = args.corpus1, args.corpus2, args.output
    lang1, lang2, shared_lang = args.lang1, args.lang2, args.shared_lang

    input_files1 = [open(corpus1 + '.' + ext) for ext in (shared_lang, lang1)]
    input_files2 = [open(corpus2 + '.' + ext) for ext in (shared_lang, lang2)]
    output_files = output1, output2, output_shared = [open(output + '.' + ext, 'w')
                                                      for ext in (lang1, lang2, shared_lang)]

    d = defaultdict(list)
    for i, (line_shared, line1) in enumerate(izip(*input_files1)):
        if line_shared.strip() and line1.strip():
            d[line_shared].append((i, line1))

    indices = []

    for line_shared, line2 in izip(*input_files2):
        lines = d[line_shared]

        if not lines:
            continue

        i, line1 = lines.pop(0)  # rather sloppy alignment
        indices.append(i)

        output1.write(line1)
        output2.write(line2)
        output_shared.write(line_shared)

    monotonicity = sum(1 for x, y in zip(indices, indices[1:]) if y > x) / len(indices)
    print 'Monotonicity rate: {}'.format(monotonicity)

    for f in output_files:
        f.close()
