#!/bin/bash
python semi_total_train_image_target_only_withinID.py --dataset office --source dslr --target webcam --method S --ul_method label --gpu_id $1 --DC
python semi_total_train_image_target_only_withinID.py --dataset office --source webcam --target dslr --method S --ul_method label --gpu_id $1 --DC
python semi_total_train_image_target_only_withinID.py --dataset office --source webcam --target amazon --method S --ul_method label --gpu_id $1 --DC