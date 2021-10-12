#!/bin/bash
python semi_total_train_image_TN.py --dataset office_home --source Real --target Art --method $2 --ul_method ENT --gpu_id $1
python semi_total_train_image_TN.py --dataset office_home --source Real --target Product --method $2 --ul_method ENT --gpu_id $1
python semi_total_train_image_TN.py --dataset office_home --source Real --target Clipart --method $2 --ul_method ENT --gpu_id $1