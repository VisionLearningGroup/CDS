#!/bin/bash
python train_image_fc_fixed_withinID.py --dataset office_home --source Art --target Product --gpu_id $1
python train_image_fc_fixed_withinID.py --dataset office_home --source Art --target Real --gpu_id $1
python train_image_fc_fixed_withinID.py --dataset office_home --source Art --target Clipart --gpu_id $1