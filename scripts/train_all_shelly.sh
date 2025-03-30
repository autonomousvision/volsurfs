#! /bin/bash

GPU_ID=0

bash scripts/train.sh shelly khady base_5 $GPU_ID
bash scripts/train.sh shelly kitten base_5 $GPU_ID
bash scripts/train.sh shelly pug base_5 $GPU_ID
bash scripts/train.sh shelly horse base_5 $GPU_ID
bash scripts/train.sh shelly fernvase base_5 $GPU_ID
bash scripts/train.sh shelly woolly base_5 $GPU_ID