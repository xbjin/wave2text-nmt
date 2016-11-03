#!/bin/sh
cd $(dirname $0)
cd ..
python3 -m translate wsd/config/wsd.yaml --train
