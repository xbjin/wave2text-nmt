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

    dev_perplexities = []
    train_perplexities = []

    with open(log_file) as f:
        for line in f:
            m = re.search('main step (\d+)', line)
            if m:
                current_step = int(m.group(1))

            m = re.search(r'eval: loss (.*)', line)
            if m:
                perplexity = float(m.group(1))
                dev_perplexities.append((current_step, perplexity))
                continue

            m = re.search(r'loss (.*)', line)
            if m:
                perplexity = float(m.group(1))
                train_perplexities.append((current_step, perplexity))
                continue

    name = 'model {}'.format(i) if len(args.log_files) > 1 else ''
    plt.plot(*zip(*dev_perplexities), label=' '.join([name, 'dev loss']))
    plt.plot(*zip(*train_perplexities), label=' '.join([name, 'train loss']))

legend = plt.legend(loc='upper center', shadow=True)

if args.output is not None:
    plt.savefig(args.output)
else:
    plt.show()
