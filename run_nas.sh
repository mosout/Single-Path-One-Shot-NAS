#!/usr/bin/env bash

set -ux

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

MODEL_NAME=nas
ITERS=30
DATASET_DIR=./time_dataset
RESULT_DIR=./exp_data
array=(1000 1500 2000 3000 4000 5000 6000 7000)
BS=256

METHOD=ours
for threshold in "${array[@]}"; do
    ONEFLOW_REMAT_SUMMARY_FILE_PREFIX=${RESULT_DIR}/$MODEL_NAME-$METHOD-$threshold ENABLE_PROFILE_FOR_DTR=0 ONEFLOW_VM_MULTI_THREAD=0 ONEFLOW_DTR_OP_TIME_DATASET=${DATASET_DIR}/${MODEL_NAME}_op_time_${BS}.json ONEFLOW_DTR_GROUP_NUM=2 python3 -u $SCRIPT_DIR/dtr_run_supernet.py $threshold $BS $ITERS
done

METHOD=dte-our-impl
for threshold in "${array[@]}"; do
    ONEFLOW_REMAT_SUMMARY_FILE_PREFIX=${RESULT_DIR}/$MODEL_NAME-$METHOD-$threshold ENABLE_PROFILE_FOR_DTR=0 ONEFLOW_VM_MULTI_THREAD=0 ONEFLOW_DTR_OP_TIME_DATASET=${DATASET_DIR}/${MODEL_NAME}_op_time_${BS}.json ONEFLOW_DTR_GROUP_NUM=1 ONEFLOW_REMAT_HEURISTIC_DTE=1 python3 -u $SCRIPT_DIR/dtr_run_supernet.py $threshold $BS $ITERS
done

METHOD=dtr-no-free
for threshold in "${array[@]}"; do
    ONEFLOW_REMAT_SUMMARY_FILE_PREFIX=${RESULT_DIR}/$MODEL_NAME-$METHOD-$threshold ENABLE_PROFILE_FOR_DTR=0 ONEFLOW_VM_MULTI_THREAD=0 ONEFLOW_DTR_OP_TIME_DATASET=${DATASET_DIR}/${MODEL_NAME}_op_time_${BS}.json ONEFLOW_DTR_GROUP_NUM=1 ONEFLOW_REMAT_HEURISTIC_DTR=1 python3 -u $SCRIPT_DIR/dtr_run_supernet.py $threshold $BS $ITERS
done

