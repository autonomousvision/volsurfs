#! /bin/bash

GPU_ID=0

bash scripts/train.sh blendernerf plushy base_5 $GPU_ID
bash scripts/train.sh blendernerf hairy_monkey base_5 $GPU_ID