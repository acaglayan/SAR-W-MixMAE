# SAR-W-MixMAE
# Copyright (c) 2025 Ali Caglayan, Nevrez Imamoglu, Toru Kouyama (AIST).
# License: MIT
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------

#!/bin/bash

source /home/<USER_ID>/miniconda/bin/activate 
conda activate /home/<USER_ID>/miniconda/envs/sarwmix

TASK_TYPE=${TASK_TYPE:-test}  # can be "pretrain", "finetune", "test"
EPOCH=${EPOCH:-64}
MODEL=${MODEL:-base}  # can be "base", "large", "huge"	${MODEL:-base}
ABLATION="${MODEL}_${EPOCH}"

WORKDIR=${PBS_O_WORKDIR:-/<PATH_2_PROJECT>/SAR-W-MixMAE}
# Directories for storing logs, models, checkpoints, etc.
MODELs_PATH="/<PATH_FOR_SAVING_MODELS>"

cd "$WORKDIR"

PT_TYPE="benv2_rand_pretrain_${MODEL}"
RUN_NAME="$PT_TYPE-$EPOCH-$ABLATION"
DATESTAMP=$(date +'%Y-%m-%d-%H-%M-%S')

NUM_GPUS_PER_NODE=2
NHOSTS=1

# get number of GPUs
GPUS_IN_ONE_NODE=$(nvidia-smi --list-gpus | wc -l)
NUM_GPU=$((NHOSTS * GPUS_IN_ONE_NODE))
echo "NUM_GPU = ${NUM_GPU}"

HOSTNAME=$(hostname)  # Set the hostlist to the current machine's hostname
export HOSTNAME      # Export it for use in the script

export OMPI_MCA_plm_rsh_disable_pty=true
export OMPI_MCA_mpi_warn_on_fork=false

export PYTHONPATH=$PYTHONPATH:/<PATH_2_PROJECT>/SAR-W-MixMAE
export PYTHONPATH=$PYTHONPATH:/<PATH_2_PROJECT>/SAR-W-MixMAE/util
export PYTHONPATH=$PYTHONPATH:/<PATH_2_PROJECT>/SAR-W-MixMAE/sarwmix

export MASTER_ADDR=$(hostname)  # Use localhost or the hostname of the machine
export MASTER_PORT=29500       # You can change this port number if necessary

# MPI options
MPIOPTS="-np $NUM_GPU -N ${NUM_GPUS_PER_NODE} -x MASTER_ADDR=${HOSTNAME} --host localhost:2"

echo "LOG_FILE = ${LOG_FILE}"
    
# Command depending on the task
if [ "$TASK_TYPE" == "pretrain" ]; then
    LOG_FILE="$WORKDIR/run_${TASK_TYPE}_$PT_TYPE-$DATESTAMP.log"
    # Run pretraining command. the parameter for inference only/reconstruction: --inference
    mpirun --oversubscribe $MPIOPTS python main_pretrain.py --dist_on_itp \
        --batch_size 32 \
        --model mixmim_${MODEL} \
        --norm_pix_loss \
        --mask_ratio 0.5 \
        --epochs 1024 \
        --accum_iter 1 \
        --warmup_epochs 40 \
        --blr 1.5e-4 --weight_decay 0.05 \
        --output_dir $MODELs_PATH/PRETr_CKPTs_LOGs/$PT_TYPE \
        --log_dir $MODELs_PATH/PRETr_CKPTs_LOGs/$PT_TYPE/log \
        --resume $MODELs_PATH/PRETr_CKPTs_LOGs/$PT_TYPE/checkpoint.pth | tee -a $LOG_FILE

elif [ "$TASK_TYPE" == "finetune" ]; then
    LOG_FILE="$WORKDIR/run_${TASK_TYPE}_$RUN_NAME-$DATESTAMP.log"
    # Run finetuning command, normal setup batch_size 128x8
    mpirun --oversubscribe $MPIOPTS python main_finetune.py --dist_on_itp \
        --batch_size 32 \
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
        --finetune $MODELs_PATH/PRETr_CKPTs_LOGs/$PT_TYPE/checkpoint_$EPOCH.pth | tee -a $LOG_FILE

elif [ "$TASK_TYPE" == "test" ]; then
    LOG_FILE="$WORKDIR/run_${TASK_TYPE}_$RUN_NAME-$DATESTAMP.log"
    # Run testing command
    echo "Running evaluation with checkpoint_best.pth" >> $LOG_FILE
    mpirun --oversubscribe $MPIOPTS python main_finetune.py --dist_on_itp \
        --batch_size 32 \
        --model mixmim_${MODEL} \
        --epochs 50 \
        --blr 5e-4 --layer_decay 0.7 \
        --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 \
        --eval \
        --dist_eval \
        --output_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --log_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --resume $MODELs_PATH/FT_CKPTs_LOGs/$PT_TYPE/$ABLATION/checkpoint_best.pth \
        --port 29528 | tee -a $LOG_FILE
    
    echo "Running evaluation with checkpoint_best_mbr.pth" >> $LOG_FILE
    mpirun --oversubscribe $MPIOPTS python main_finetune.py --dist_on_itp \
        --batch_size 32 \
        --model mixmim_${MODEL} \
        --epochs 50 \
        --blr 5e-4 --layer_decay 0.7 \
        --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 \
        --eval \
        --dist_eval \
        --output_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --log_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --resume $MODELs_PATH/FT_CKPTs_LOGs/$PT_TYPE/$ABLATION/checkpoint_best_mbr.pth \
        --port 29528 | tee -a $LOG_FILE
    
    echo "Running evaluation with checkpoint_best_default.pth" >> $LOG_FILE
    mpirun --oversubscribe $MPIOPTS python main_finetune.py --dist_on_itp \
        --batch_size 32 \
        --model mixmim_${MODEL} \
        --epochs 50 \
        --blr 5e-4 --layer_decay 0.7 \
        --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 \
        --eval \
        --dist_eval \
        --output_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --log_dir $MODELs_PATH/TESTS/$PT_TYPE/$ABLATION \
        --resume $MODELs_PATH/FT_CKPTs_LOGs/$PT_TYPE/$ABLATION/checkpoint_best_default.pth \
        --port 29528 | tee -a $LOG_FILE
fi
    


conda deactivate
