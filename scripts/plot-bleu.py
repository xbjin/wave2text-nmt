#!/usr/bin/env python3

import argparse
from matplotlib import pyplot as plt
import re

parser = argparse.ArgumentParser()
parser.add_argument('log_files', nargs='+')
parser.add_argument('--output')

args = parser.parse_args()

for i, log_file in enumerate(args.log_files, 1):
    current_step = 0

    bleu_scores = []

    with open(log_file) as f:
        for line in f:
            m = re.search('main step (\d+)', line)
            if m:
                current_step = int(m.group(1))

            m = re.search(r'score=(\d+\.\d+)', line)
            if m:
                bleu_score = float(m.group(1))
                bleu_scores.append((current_step, bleu_score))
                continue

    # x, y = list(zip(*bleu_scores))
    # x = np.array(x)
    #
    # from scipy.interpolate import spline
    # import numpy as np
    #
    # new_x = np.linspace(x.min(), x.max(), 100)
    # new_y = spline(x, y, new_x)
    # plt.plot(new_x, new_y)

    name = 'model {}'.format(i) if len(args.log_files) > 1 else ''
    plt.plot(*zip(*bleu_scores), label=' '.join([name, 'dev BLEU']))

legend = plt.legend(loc='upper center', shadow=True)

if args.output is not None:
    plt.savefig(args.output)
else:
    plt.show()
