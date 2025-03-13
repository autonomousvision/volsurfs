#! /bin/bash
# usage: bash train.sh <DATASET_NAME> <SCENE_NAME> <CUDA_DEVICE_ID>

DATASET_NAME=$1
SCENE_NAME=$2
EXP_NAME=$3
GPU_ID=$4

echo "DATASET_NAME: $DATASET_NAME"
echo "SCENE_NAME: $SCENE_NAME"
echo "EXP_NAME: $EXP_NAME"
echo "GPU_ID: $GPU_ID"

# surf
bash bash/surf.sh $DATASET_NAME $SCENE_NAME base $GPU_ID

# offsets_surfs (train main SDF and offsets jointly), bake meshes
bash bash/offsets_surfs.sh $DATASET_NAME $SCENE_NAME "$EXP_NAME" $GPU_ID

# volsurfs (train neural textures on meshes, bake textures)
bash bash/volsurfs.sh $DATASET_NAME $SCENE_NAME "$EXP_NAME" offsets_surfs "$EXP_NAME" $GPU_ID