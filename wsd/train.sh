#!/bin/sh
cd $(dirname $0)
cd ..
python2 -m translate wsd/config/wsd.yaml --train
