#!/bin/bash
python train_image_fc_fixed_withinID.py --dataset office_home --source Clipart --target Art --gpu_id $1
python train_image_fc_fixed_withinID.py --dataset office_home --source Clipart --target Real --gpu_id $1
python train_image_fc_fixed_withinID.py --dataset office_home --source Clipart --target Product --gpu_id $1