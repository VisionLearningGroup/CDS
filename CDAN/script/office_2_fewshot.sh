#!/bin/bash
python semi_total_train_image_fixed_fewshot.py --dataset office --source dslr --target webcam --method $2 --ul_method ENT --gpu_id $1
python semi_total_train_image_fixed_fewshot.py --dataset office --source webcam --target dslr --method $2 --ul_method ENT --gpu_id $1
python semi_total_train_image_fixed_fewshot.py --dataset office --source webcam --target amazon --method $2 --ul_method ENT --gpu_id $1