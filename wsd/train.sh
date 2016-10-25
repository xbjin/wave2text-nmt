#!/bin/sh
cd $(dirname $0)
cd ..
python2 -m translate config/wsd.yaml --train
