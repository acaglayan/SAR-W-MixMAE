# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# batch job with qsub command for ABCI: https://docs.abci.ai/v3/en/job-execution/
# ------------------------------------------------------------------------

#!/bin/bash

#PBS -q rt_HF
#PBS -l select=1:ncpus=192
#PBS -l walltime=24:00:00
#PBS -P <GROUP_ID>
#PBS -j oe
#PBS -o run_output.log

source /etc/profile.d/modules.sh
module load hpcx/2.20
module load cuda/12.6/12.6.1
module load cudnn/9.5/9.5.1
module load nccl/2.23/2.23.4-1

source /home/<USER_ID>/.bashrc
source /home/<USER_ID>/anaconda3/bin/activate /home/<USER_ID>/anaconda3/envs/sarwmix

TASK_TYPE=${TASK_TYPE:-pretrain}  # can be "pretrain", "finetune", "test"
EPOCH=${EPOCH:-64}
MODEL=${MODEL:-base}  # can be "base", "large", "huge"	${MODEL:-base}
ABLATION="${MODEL}_${EPOCH}"

cd $PBS_O_WORKDIR

PT_TYPE="benv2_rand_pretrain_${MODEL}"
RUN_NAME="$PT_TYPE-$EPOCH-$ABLATION"
DATESTAMP=$(date +'%Y-%m-%d-%H-%M-%S')

# Directories for storing logs, models, checkpoints, etc.
MODELs_PATH="/<PATH_FOR_SAVING_MODELS>"


# detect GPU type, V100 or A100 or H200
#GPU_INFO=$(nvidia-smi --query-gpu=gpu_name --format=csv)
GPU_INFO=$(nvidia-smi --query-gpu=gpu_name --format=csv | tail -n +2)
if [[ $GPU_INFO =~ "H200" ]]; then
    NUM_GPUS_PER_NODE=8
elif [[ $GPU_INFO =~ "V100" ]]; then
    NUM_GPUS_PER_NODE=4
elif [[ $GPU_INFO =~ "A100" ]]; then
    NUM_GPUS_PER_NODE=8
else
    readonly PROC_ID=$!
    kill ${PROC_ID}
fi

# get number of GPUs
GPUS_IN_ONE_NODE=$(nvidia-smi --list-gpus | wc -l)
echo "GPUS_IN_ONE_NODE = ${GPUS_IN_ONE_NODE}"
# NHOSTS=${PBS_NUM_NODES:-1}  # Default to 1 node if NHOSTS is undefined
# echo "NHOSTS = ${NHOSTS}"
NHOSTS=$(cat $PBS_NODEFILE | uniq | wc -l)
echo "NHOSTS = ${NHOSTS}"

NUM_GPU=$(expr ${NHOSTS} \* ${GPUS_IN_ONE_NODE})
echo "NUM_GPU = ${NUM_GPU}"


# MPI options
MPIOPTS="-np $NUM_GPU -N ${NUM_GPUS_PER_NODE} -x MASTER_ADDR=${HOSTNAME} -hostfile $PBS_NODEFILE"

cd /home/<USER_ID>/Works/SAR-W-MixMAE/

work_path=/home/<USER_ID>/Works/SAR-W-MixMAE/

# Command depending on the task
if [ "$TASK_TYPE" == "pretrain" ]; then
    LOG_FILE="$PBS_O_WORKDIR/run_${TASK_TYPE}_$PT_TYPE-$DATESTAMP.log"
    # Run pretraining command. the parameter for inference only/reconstruction: --inference
    mpirun --oversubscribe $MPIOPTS python main_pretrain.py --dist_on_itp \
        --batch_size 256 \
        --model mixmim_${MODEL} \
        --norm_pix_loss \
        --mask_ratio 0.5 \
        --epochs 1024 \
        --accum_iter 1 \
        --warmup_epochs 40 \
        --blr 1.5e-4 --weight_decay 0.05 \
        --output_dir $MODELs_PATH/PRETr_CKPTs_LOGs/$PT_TYPE \
        --log_dir $MODELs_PATH/PRETr_CKPTs_LOGs/$PT_TYPE/log \
        --resume $MODELs_PATH/PRETr_CKPTs_LOGs/$PT_TYPE/checkpoint.pth >> $LOG_FILE 2>&1

elif [ "$TASK_TYPE" == "finetune" ]; then
    LOG_FILE="$PBS_O_WORKDIR/run_${TASK_TYPE}_$RUN_NAME-$DATESTAMP.log"
    # Run finetuning command, normal setup batch_size 128x8
    mpirun --oversubscribe $MPIOPTS python main_finetune.py --dist_on_itp \
        --batch_size 128 \
        --model mixmim_${MODEL} \
        --epochs 50 \
        --warmup_epochs 5 \
        --blr 5e-4 --layer_decay 0.7 \
        --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 \
        --dist_eval \
        --output_dir $MODELs_PATH/FT_CKPTs_LOGs/$PT_TYPE/$ABLATION \
        --log_dir $MODELs_PATH/FT_CKPTs_LOGs/$PT_TYPE/$ABLATION/log \
        --resume $MODELs_PATH/FT_CKPTs_LOGs/$PT_TYPE/$ABLATION/checkpoint.pth \
        --port 29528 \
        --finetune $MODELs_PATH/PRETr_CKPTs_LOGs/$PT_TYPE/checkpoint_$EPOCH.pth >> $LOG_FILE 2>&1

elif [ "$TASK_TYPE" == "test" ]; then
    LOG_FILE="$PBS_O_WORKDIR/run_${TASK_TYPE}_$RUN_NAME-$DATESTAMP.log"
    # Run testing command
    echo "Running evaluation with checkpoint_best.pth" >> $LOG_FILE
    mpirun --oversubscribe $MPIOPTS python main_finetune.py --dist_on_itp \
        --batch_size 1024 \
        --model mixmim_${MODEL} \
        --epochs 50 \
        --blr 5e-4 --layer_decay 0.7 \
        --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 \
        --eval \
        --dist_eval \
        --output_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --log_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --resume $MODELs_PATH/FT_CKPTs_LOGs/$PT_TYPE/$ABLATION/checkpoint_best.pth \
        --port 29528 >> $LOG_FILE 2>&1
    
    echo "Running evaluation with checkpoint_best_mbr.pth" >> $LOG_FILE
    mpirun --oversubscribe $MPIOPTS python main_finetune.py --dist_on_itp \
        --batch_size 1024 \
        --model mixmim_${MODEL} \
        --epochs 50 \
        --blr 5e-4 --layer_decay 0.7 \
        --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 \
        --eval \
        --dist_eval \
        --output_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --log_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --resume $MODELs_PATH/FT_CKPTs_LOGs/$PT_TYPE/$ABLATION/checkpoint_best_mbr.pth \
        --port 29528 >> $LOG_FILE 2>&1
    
    echo "Running evaluation with checkpoint_best_default.pth" >> $LOG_FILE
    mpirun --oversubscribe $MPIOPTS python main_finetune.py --dist_on_itp \
        --batch_size 1024 \
        --model mixmim_${MODEL} \
        --epochs 50 \
        --blr 5e-4 --layer_decay 0.7 \
        --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 \
        --eval \
        --dist_eval \
        --output_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --log_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --resume $MODELs_PATH/FT_CKPTs_LOGs/$PT_TYPE/$ABLATION/checkpoint_best_default.pth \
        --port 29528 >> $LOG_FILE 2>&1
fi



source /home/<USER_ID>/anaconda3/bin/deactivate
