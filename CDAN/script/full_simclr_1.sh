#!/bin/bash
python train_image_fc_fixed_simclr.py --dataset office_home --source Real --target Clipart --gpu_id 0 --DC
python train_image_fc_fixed_simclr.py --dataset office_home --source Clipart --target Real --gpu_id 0
python train_image_fc_fixed_simclr.py --dataset office_home --source Art --target Product --gpu_id 0 --DC