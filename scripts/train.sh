#!/usr/bin/env bash

tag='tmp'

log='ckpts/'$tag'.log'
rm $log
nohup python -u main.py --tag=$tag > $log 2>&1 &
sleep 2
tail -f $log
