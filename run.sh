#!/bin/bash
# Note: This type of for-loop requires bash shell, not sh

RESULT_DIR="./output/results"

for rep in {1..10}
do
  for args in "alpha=1. sparsity=0.1 n_train_samp="{500,750,1000,1250,1500}
  do 
    echo `date +"%Y-%m-%d %T"` "with np_seed=${rep} ${args}"
    python exp.py -t ${RESULT_DIR} -l INFO with np_seed=$rep $args
  done
done
