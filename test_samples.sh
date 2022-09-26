#!/usr/bin/env bash
set -x
for (( i = 2; i <= 4; i++ )) ;
do
  echo "group: ${i}"
  python test_ytvos_samples.py --dataset_file mini-ytvos --group=${i}
done
for (( i = 2; i <= 4; i++ )) ;
do
  echo "group: ${i}"
  python test_ytvos_samples.py --dataset_file sailvos --group=${i}
done

#python3 test.py --cat_id 5 --dataset_file gygo