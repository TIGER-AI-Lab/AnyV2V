#!/bin/bash
#source /home/YourName/miniconda3/etc/profile.d/conda.sh  #<-- change this to your own miniconda path
conda activate anyv2v-i2vgen-xl

cd ..
python run_group_ddim_inversion.py \
--template_config "configs/group_ddim_inversion/template.yaml" \
--configs_json "configs/group_ddim_inversion/group_config.json"