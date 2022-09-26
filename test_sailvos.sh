#!/usr/bin/env bash
set -x
for (( i = 1; i <= 3; i++ )) ;
do
  python test.py --cat_id=${i} --dataset_file sailvos
done