#!/bin/bash
python train_image_fc_fixed_simclr.py --dataset office_home --source Art --target Real --gpu_id 1 --DC
python train_image_fc_fixed_simclr.py --dataset office_home --source Art --target Clipart --gpu_id 1 --DC
python train_image_fc_fixed_simclr.py --dataset office_home --source Product --target Clipart --gpu_id 1 --DC