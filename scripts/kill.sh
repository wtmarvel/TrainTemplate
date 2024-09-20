#!/usr/bin/env bash

#name='spawn_main'
#kill $(ps aux | grep $name | grep -v grep | awk '{print $2}')
#kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')

pkill -f 'spawn_main|main_multi_nodes|wt/.pycharm_helpers'

