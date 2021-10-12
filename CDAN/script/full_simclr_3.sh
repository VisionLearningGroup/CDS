#!/bin/bash
python train_image_fc_fixed_simclr.py --dataset office_home --source Product --target Real --gpu_id 0 --DC
python train_image_fc_fixed_simclr.py --dataset office_home --source Real --target Product --gpu_id 0 --DC
python train_image_fc_fixed_simclr.py --dataset office_home --source Real --target Art --gpu_id 0 --DC