#!/usr/bin/env bash
set -x
for (( i = 1; i <= 11; i++ )) ;
do
  python3 test.py --cat_id=${i} --dataset_file gygo
done
#python3 test.py --cat_id 5 --dataset_file gygo