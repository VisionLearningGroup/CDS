#!/bin/bash
python semi_total_train_image_withinID.py --dataset office --source amazon --target dslr --method CDAN+E --ul_method ENT --gpu_id $1
python semi_total_train_image_withinID.py --dataset office --source amazon --target webcam --method CDAN+E --ul_method ENT --gpu_id $1
python semi_total_train_image_withinID.py --dataset office --source dslr --target amazon --method CDAN+E--ul_method ENT --gpu_id $1