#!/bin/bash
python semi_total_train_image_target_only_from_singledomain_simclr.py --dataset office --source webcam --method S --ul_method label --gpu_id $1
python semi_total_train_image_target_only_from_singledomain_simclr.py --dataset office --source amazon --method S --ul_method label --gpu_id $1
python semi_total_train_image_target_only_from_singledomain_simclr.py --dataset office --source dslr --method S --ul_method label --gpu_id $1