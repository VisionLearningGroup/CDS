#!/bin/bash
python semi_total_train_image_target_only.py --dataset office_home --source Clipart --target Art --method $2 --ul_method label --gpu_id $1
python semi_total_train_image_target_only.py --dataset office_home --source Clipart --target Product --method $2 --ul_method label --gpu_id $1
python semi_total_train_image_target_only.py --dataset office_home --source Clipart --target Real --method $2 --ul_method label --gpu_id $1