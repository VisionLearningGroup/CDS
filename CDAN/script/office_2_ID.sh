#!/bin/bash
python semi_total_train_image_ID.py --dataset office --source dslr --target webcam --method CDAN+E --ul_method ENT --gpu_id $1
python semi_total_train_image_ID.py --dataset office --source webcam --target dslr --method CDAN+E --ul_method ENT --gpu_id $1
python semi_total_train_image_ID.py --dataset office --source webcam --target amazon --method CDAN+E --ul_method ENT --gpu_id $1