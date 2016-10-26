#!/bin/sh
cd $(dirname $0)
cd ..
./scripts/prepare-data.py wsd/corpus/semcor12 lem syn wsd/data/ --dev-size 1000 --test-size 1000 --no-tokenize --shuffle --vocab-size 0

