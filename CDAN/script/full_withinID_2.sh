#!/bin/bash
python train_image_fc_fixed_withinID.py --dataset office_home --source Product --target Clipart --gpu_id $1
python train_image_fc_fixed_withinID.py --dataset office_home --source Product --target Real --gpu_id $1
python train_image_fc_fixed_withinID.py --dataset office_home --source Product --target Art --gpu_id $1