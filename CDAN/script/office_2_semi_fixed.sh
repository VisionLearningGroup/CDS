#!/bin/bash
python semi_total_train_image_fixed.py --dataset office --source dslr --target webcam --method S --ul_method ENT --gpu_id $1
python semi_total_train_image_fixed.py --dataset office --source webcam --target dslr --method S --ul_method ENT --gpu_id $1
python semi_total_train_image_fixed.py --dataset office --source webcam --target amazon --method S --ul_method ENT --gpu_id $1