#!/bin/bash
python train_image_fc_fixed_withinID.py --dataset office_home --source Real --target Product --gpu_id $1
python train_image_fc_fixed_withinID.py --dataset office_home --source Real --target Art --gpu_id $1
python train_image_fc_fixed_withinID.py --dataset office_home --source Real --target Clipart --gpu_id $1