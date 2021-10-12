#!/bin/bash
python semi_total_train_image_target_only.py --dataset office --source amazon --target dslr --method $2 --ul_method ENT --gpu_id $1
python semi_total_train_image_target_only.py --dataset office --source amazon --target webcam --method $2 --ul_method ENT --gpu_id $1
python semi_total_train_image_target_only.py --dataset office --source dslr --target amazon --method $2 --ul_method ENT --gpu_id $1