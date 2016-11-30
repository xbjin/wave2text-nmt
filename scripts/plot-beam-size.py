#!/usr/bin/env python3

import argparse
from translate.utils import bleu_score
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('ref')
parser.add_argument('mt', nargs='+')
parser.add_argument('--plots', type=int, default=1)
parser.add_argument('--scripts', default='scripts')
parser.add_argument('--output')

args = parser.parse_args()

bleus = []

with open(args.ref) as f:
    references = [line.strip('\n') for line in f]

for filename in args.mt:
    with open(filename) as f:
        hypotheses = [line.strip('\n') for line in f]

    bleu, _ = bleu_score(hypotheses, references, args.scripts)
    bleus.append(bleu)

assert len(bleus) % args.plots == 0

n = len(bleus) // args.plots
bleus = [bleus[i * n: (i + 1) * n] for i in range(args.plots)]

for bleus_ in bleus:
    plt.plot(range(1, n + 1), bleus_)

if args.output is not None:
    plt.savefig(args.output)
else:
    plt.show()
