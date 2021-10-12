#!/bin/bash
python semi_total_train_image_target_only_instance_only.py --dataset office_home --source Art --target Clipart --method $2 --ul_method label --gpu_id $1
python semi_total_train_image_target_only_instance_only.py --dataset office_home --source Art --target Product --method $2 --ul_method label --gpu_id $1
python semi_total_train_image_target_only_instance_only.py --dataset office_home --source Art --target Real --method $2 --ul_method label --gpu_id $1