#!/usr/bin/env bash

root_dir=experiments/character_level
scripts/prepare-data.py data/raw/btec.fr-en/btec-train fr en ${root_dir}/data --lowercase --character-level en --min 1 --max 0
scripts/prepare-data.py data/raw/btec.fr-en/btec-test1 fr en mref.en ${root_dir}/data --lowercase --min 1 --max 0 --mode prepare --output test1
scripts/prepare-data.py data/raw/btec.fr-en/btec-test2 fr en mref.en ${root_dir}/data --lowercase --min 1 --max 0 --mode prepare --output test2
scripts/prepare-data.py data/raw/btec.fr-en/btec-dev-concat fr en ${root_dir}/data --lowercase --min 1 --max 0 --mode prepare --output dev