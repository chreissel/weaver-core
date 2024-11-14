#!/bin/bash

set -x
echo "args: $@"
# set the dataset dir via `DATADIR_JetClass`
DATADIR=/n/home03/creissel/weaver-core/TEST/

# set a comment via `COMMENT`
suffix="test"

# set the number of gpus for DDP training via `DDP_NGPUS`
CMD="weaver"

dataopts="--num-workers 1 --fetch-step 1.0 --in-memory"
out_dim=8

bs=512
epochs=1
samples_per_ep=$(( $bs * 1000 ))
samples_per_ep_val=$(( $bs * 250 ))

model="ParT"
modelopts="weaver/nn/model/ParT_embedder.py --optimizer-option weight_decay 0.01 --optimizer adamW"
sched_opts="--lr-scheduler flat+cos --start-lr 1e-3"
batchopts="--num-epochs $epochs --batch-size $bs --samples-per-epoch $samples_per_ep --samples-per-epoch-val $samples_per_ep_val" #negative value to force loading entire val set

suffix=${model}

$CMD \
    --data-train \
    "TTBar:${DATADIR}/Tbqq/Tbqq_part0000_test.parquet" \
    "QCD:${DATADIR}/QCD/*.parquet" \
    "Zqq:${DATADIR}/Zqq/*.parquet" \
    --data-test \
    "TTBar:${DATADIR}/Tbqq/Tbqq_part0001_test.parquet" \
    "QCD:${DATADIR}/QCD/*.parquet" \
    "Zqq:${DATADIR}/Zqq/*.parquet" \
    --data-config data/CMS_monojet.yaml --network-config $modelopts \
    --model-prefix trainings/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
    $dataopts $batchopts $sched_opts --gpus 0 \
    --predict-output pred.parquet \
    --tensorboard JetClass_${FEATURE_TYPE}_${suffix} --data-fraction 1.0 \
    --contrastive-mode \
    --coordinates ptetaphi
