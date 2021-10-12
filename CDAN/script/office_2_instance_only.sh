#!/bin/bash
python semi_total_train_image_target_only_instance_only.py --dataset office --source dslr --target webcam --method S --ul_method label --gpu_id $1
python semi_total_train_image_target_only_instance_only.py --dataset office --source webcam --target dslr --method S --ul_method label --gpu_id $1
python semi_total_train_image_target_only_instance_only.py --dataset office --source webcam --target amazon --method S --ul_method label --gpu_id $1