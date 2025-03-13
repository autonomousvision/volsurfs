#! /bin/bash

# usage:
# bash nerf.sh <DATASET_NAME> <SCENE_NAME> <EXP_NAME> <CUDA_DEVICE_ID>

#############################

DATASET_NAME=$1
SCENE_NAME=$2
METHOD_NAME=nerf
EXP_NAME=$3
GPU_ID=$4

echo "DATASET_NAME: $DATASET_NAME"
echo "SCENE_NAME: $SCENE_NAME"
echo "METHOD_NAME: $METHOD_NAME"
echo "EXP_NAME: $EXP_NAME"

EVAL_TRAIN=false
EVAL_TEST=true

# set flags
EVAL_ARGS=""
if [ "$EVAL_TRAIN" = true ]; then
    EVAL_ARGS="$EVAL_ARGS --eval_train"
fi
if [ "$EVAL_TEST" = true ]; then
    EVAL_ARGS="$EVAL_ARGS --eval_test"
fi

RUN_TRAINING=true
RUN_EVALUATION=false
REMOVE_RENDERS=false

BASE="--method_name $METHOD_NAME --dataset $DATASET_NAME --scene $SCENE_NAME --exp_name $EXP_NAME"
RUNS_PATH=$(grep 'runs:' config/paths_config.cfg | sed 's/[^/]*//; s/[\{\} "]//g')
echo "RUNS_PATH: $RUNS_PATH"

echo "$METHOD_NAME with config $EXP_NAME"

export WANDB__SERVICE_WAIT=300
export CUDA_VISIBLE_DEVICES=$GPU_ID
export OMP_NUM_THREADS=16

# if $RUN_TRAINING is set
if [ "$RUN_TRAINING" = true ]; then

    echo "RUNNING TRAINING"

    echo "python ./volsurfs_py/trainer.py $BASE"
    python ./volsurfs_py/trainer.py $BASE --train $EVAL_ARGS --keep_last_checkpoint_only

fi

if [ "$REMOVE_RENDERS" = true ]; then

    echo "REMOVING RENDERS"

    # remove renders folder
    RUN_ID=""
    for d in $(ls -t $RUNS_PATH/$METHOD_NAME/$EXP_NAME/$SCENE_NAME); do
        RUN_ID="$d"
        break
    done

    RUN_PATH="$RUNS_PATH/$METHOD_NAME/$EXP_NAME/$SCENE_NAME/$RUN_ID"
    echo "RUN_PATH: $RUN_PATH"

    # find last checkpoint in RUN_PATH
    CKPT_NR=""
    for d in $(ls -t $RUN_PATH); do
        # if folder name if "config" or "wandb_imgs", skip
        if [ "$d" = "config" ] || [ "$d" = "wandb_imgs" ] || [ "$d" = "results" ]; then
            continue
        fi
        CKPT_NR="$d"
        break
    done

    # check if CKPT_NR is set
    if [ -z "$CKPT_NR" ]; then
        echo "CKPT_NR not found"
        exit 1
    else
        echo "CKPT_NR: $CKPT_NR"
    fi

    RENDERS_PATH="$RUN_PATH/$CKPT_NR/renders"

    # if RENDERS_PATH exists, remove it
    if [ -d "$RENDERS_PATH" ]; then
        echo "RENDERS_PATH exists, removing it"
        rm -r $RENDERS_PATH
    fi
fi

# if $RUN_EVALUATION is set
if [ "$RUN_EVALUATION" = true ]; then

    echo "RUNNING EVALUATION"

    RUN_ID=""
    for d in $(ls -t $RUNS_PATH/$METHOD_NAME/$EXP_NAME/$SCENE_NAME); do
        RUN_ID="$d"
        break
    done

    # check if RUN_ID is set
    if [ -z "$RUN_ID" ]; then
        echo "RUN_ID not found"
        exit 1
    else
        echo "RUN_ID: $RUN_ID"
    fi

    python ./volsurfs_py/trainer.py $BASE $EVAL_ARGS --run_id $RUN_ID
fi