#!/bin/bash

set -x
echo "args: $@"
# set the dataset dir via `DATADIR_JetClass`
DATADIR=/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass

# set a comment via `COMMENT`
suffix="test"

# set the number of gpus for DDP training via `DDP_NGPUS`
CMD="weaver"

dataopts="--num-workers 1 --fetch-step 0.01"
out_dim=8

bs=512
epochs=50
samples_per_ep=$(( $bs * 1000 ))
samples_per_ep_val=$(( $bs * 250 ))

model="ParT"
modelopts="weaver/nn/model/ParT_embedder.py --use-amp --optimizer-option weight_decay 0.01 --out-dim $out_dim --optimizer adamW"
sched_opts="--lr-scheduler flat+cos --start-lr 1e-3"
batchopts="--num-epochs $epochs --batch-size $bs --samples-per-epoch $samples_per_ep --samples-per-epoch-val $samples_per_ep_val" #negative value to force loading entire val set

suffix=${model}

$CMD \
    --data-train \
    "TTBar:${DATADIR}/train_100M/TTBar_*.root" \
    "WToQQ:${DATADIR}/train_100M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/train_100M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/train_100M/ZJetsToNuNu_*.root" \
    --data-val \
    "TTBar:${DATADIR}/val_5M/TTBar_*.root" \
    "WToQQ:${DATADIR}/val_5M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/val_5M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/val_5M/ZJetsToNuNu_*.root" \
    --data-test \
    "TTBar:${DATADIR}/test_10M/TTBar_100.root" \
    "WToQQ:${DATADIR}/test_10M/WToQQ_100.root" \
    "ZToQQ:${DATADIR}/test_10M/ZToQQ_100.root" \
    "ZJetsToNuNu:${DATADIR}/test_10M/ZJetsToNuNu_100.root" \
    "HToBB:${DATADIR}/test_10M/HToBB_100.root" \
    "HToCC:${DATADIR}/test_10M/HToCC_100.root" \
    "HToGG:${DATADIR}/test_10M/HToGG_100.root" \
    "HToWW4Q:${DATADIR}/test_10M/HToWW4Q_100.root" \
    --data-config data/JetClass.yaml --network-config $modelopts \
    --model-prefix trainings/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
    $dataopts $batchopts $sched_opts --gpus 0\
    --predict-output pred.root \
    --tensorboard JetClass_${FEATURE_TYPE}_${suffix} --data-fraction 1.0 \
    --contrastive-mode
