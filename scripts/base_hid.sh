#!/bin/bash

HID_FEATS=${1:-4}

COMMAND="python ../base_run.py"
NPROCS=4
MODEL="sage"
BATCH_SIZE=512
GRAPH_NAME="ogbn-products"
TOTAL_EPOCHS=6

$COMMAND --model $MODEL --nprocs $NPROCS --hid_feats $HID_FEATS --batch_size $BATCH_SIZE --graph_name $GRAPH_NAME --total_epochs $TOTAL_EPOCHS
