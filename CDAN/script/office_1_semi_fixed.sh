#!/bin/bash
python semi_total_train_image_fixed.py --dataset office --source amazon --target dslr --method S --ul_method ENT --gpu_id $1
python semi_total_train_image_fixed.py --dataset office --source amazon --target webcam --method S --ul_method label --gpu_id $1
python semi_total_train_image_fixed.py --dataset office --source dslr --target amazon --method S --ul_method ENT --gpu_id $1