#!/usr/bin/env python3

import argparse
from matplotlib import pyplot as plt
import re

parser = argparse.ArgumentParser()
parser.add_argument('log_files', nargs='+')

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

            m = re.search(r'eval: perplexity (.*)', line)
            if m:
                perplexity = float(m.group(1))
                dev_perplexities.append((current_step, perplexity))
                continue

            m = re.search(r'perplexity (.*)', line)
            if m:
                perplexity = float(m.group(1))
                train_perplexities.append((current_step, perplexity))
                continue

    plt.plot(*zip(*dev_perplexities), label='model {} dev'.format(i))
    plt.plot(*zip(*train_perplexities), label='model {} train'.format(i))

legend = plt.legend(loc='upper center', shadow=True)

plt.show()