#!/bin/bash
python semi_total_train_image.py --dataset domainnet --source real  --target painting --method CDAN+E --ul_method ENT --gpu_id 0
python semi_total_train_image.py --dataset domainnet --source painting  --target real --method CDAN+E --ul_method ENT --gpu_id 0