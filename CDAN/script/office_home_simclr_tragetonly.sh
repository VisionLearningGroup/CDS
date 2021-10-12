#!/bin/bash
python semi_total_train_image_target_only_from_singledomain_simclr.py --dataset office_home --source Clipart --method S --ul_method label --gpu_id $1
python semi_total_train_image_target_only_from_singledomain_simclr.py --dataset office_home --source Real Product --method S --ul_method label --gpu_id $1
python semi_total_train_image_target_only_from_singledomain_simclr.py --dataset office_home --source Product --method S --ul_method label --gpu_id $1
python semi_total_train_image_target_only_from_singledomain_simclr.py --dataset office_home --source Art --method S --ul_method label --gpu_id $1