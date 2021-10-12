#!/bin/bash
python semi_total_train_image_target_only_simclr.py --dataset office_home --source Art --target Clipart --method S --ul_method label --gpu_id $1
python semi_total_train_image_target_only_simclr.py --dataset office_home --source Art --target Product --method S --ul_method label --gpu_id $1
python semi_total_train_image_target_only_simclr.py --dataset office_home --source Art --target Real --method S --ul_method label --gpu_id $1