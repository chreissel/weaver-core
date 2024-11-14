#!/bin/bash
#SBATCH --job-name=weaver_JetClass
#SBATCH --partition=gpu_test
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --chdir=/n/home03/creissel/weaver-core
#SBATCH --output=slurm_monitoring/%x-%j.out

source ~/.bashrc
source /n/home03/creissel/miniforge3/etc/profile.d/conda.sh
conda activate weaver

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
epochs=40
samples_per_ep=$(( $bs * 1000 ))
samples_per_ep_val=$(( $bs * 250 ))

model="ParT"
modelopts="weaver/nn/model/ParT_embedder.py --use-amp --optimizer-option weight_decay 0.01 --out-dim $out_dim --optimizer adamW"
sched_opts="--lr-scheduler flat+cos --start-lr 1e-3"
batchopts="--num-epochs $epochs --batch-size $bs --samples-per-epoch $samples_per_ep --samples-per-epoch-val $samples_per_ep_val" #negative value to force loading entire val set

suffix=${model}

srun $CMD \
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
    "TTBar:${DATADIR}/test_20M/TTBar_100.root" \
    "HToBB:${DATADIR}/test_20M/HToBB_100.root" \
    "HToCC:${DATADIR}/test_20M/HToCC_100.root" \
    "HToGG:${DATADIR}/test_20M/HToGG_100.root" \
    "HToWW2Q1L:${DATADIR}/test_20M/HToWW2Q1L_100.root" \
    "HToWW4Q:${DATADIR}/test_20M/HToWW4Q_100.root" \
    "TTBar:${DATADIR}/test_20M/TTBar_100.root" \
    "TTBarLep:${DATADIR}/test_20M/TTBarLep_100.root" \
    "WToQQ:${DATADIR}/test_20M/WToQQ_100.root" \
    "ZToQQ:${DATADIR}/test_20M/ZToQQ_100.root" \
    "ZJetsToNuNu:${DATADIR}/test_20M/ZJetsToNuNu_100.root" \
    --data-config data/JetClass.yaml --network-config $modelopts \
    --model-prefix trainings/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
    $dataopts $batchopts $sched_opts --gpus 0\
    --predict-output pred.root \
    --tensorboard JetClass_${FEATURE_TYPE}_${suffix} --data-fraction 1.0 \
    --contrastive-mode
