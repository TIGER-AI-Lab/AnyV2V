#!/bin/bash
source /home/YourName/miniconda3/etc/profile.d/conda.sh #<-- change this to your own miniconda path
conda activate anyv2v-i2vgen-xl

cd ..
python run_group_pnp_edit.py \
--template_config "configs/group_pnp_edit/template.yaml" \
--configs_json "configs/group_pnp_edit/group_config.json"