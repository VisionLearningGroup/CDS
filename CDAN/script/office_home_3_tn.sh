#!/bin/bash
python semi_total_train_image_TN.py --dataset office_home --source Product --target Art --method $2 --ul_method ENT --gpu_id $1
python semi_total_train_image_TN.py --dataset office_home --source Product --target Clipart --method $2 --ul_method ENT --gpu_id $1
python semi_total_train_image_TN.py --dataset office_home --source Product --target Real --method $2 --ul_method ENT --gpu_id $1