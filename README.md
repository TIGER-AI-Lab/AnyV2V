# <img src="https://tiger-ai-lab.github.io/AnyV2V/static/images/icon.png" width="30"/> AnyV2V
[![arXiv](https://img.shields.io/badge/arXiv-2403.14468-b31b1b.svg)](https://arxiv.org/abs/2403.14468)
[![contributors](https://img.shields.io/github/contributors/TIGER-AI-Lab/AnyV2V)](https://github.com/TIGER-AI-Lab/AnyV2V/graphs/contributors)
[![open issues](https://isitmaintained.com/badge/open/TIGER-AI-Lab/AnyV2V.svg)](https://github.com/TIGER-AI-Lab/AnyV2V/issues)
[![pull requests](https://img.shields.io/github/issues-pr/TIGER-AI-Lab/AnyV2V?color=0088ff)](https://github.com/TIGER-AI-Lab/AnyV2V/pulls)
[![license](https://img.shields.io/github/license/TIGER-AI-Lab/AnyV2V.svg)](https://github.com/TIGER-AI-Lab/AnyV2V/blob/main/LICENSE)

AnyV2V: A Plug-and-Play Framework For Any Video-to-Video Editing Tasks

<div align="center">
<img src="https://github.com/TIGER-AI-Lab/AnyV2V/blob/gh-pages/static/images/banner.png" width="70%">
</div>

AnyV2V is a tuning-free framework to achieve high appearance and temporal consistency in video editing.
- can seamlessly build on top of advanced image editing methods to perform diverse types of editing
- robust performance on the four tasks:
  - prompt-based editing
  - reference-based style transfer
  - subject-driven editing
  - identity manipulation

<div align="center">
 <a href = "https://tiger-ai-lab.github.io/AnyV2V/">[🌐 Project Page]</a> <a href = "https://arxiv.org/abs/2403.14468">[📄 Arxiv]</a> 
</div>

## 📰 News
* 2024 Mar 22: Code released for AnyV2V(i2vgen-xl).
* 2024 Mar 21: Our paper is featured on [Huggingface Daily Papers](https://huggingface.co/papers/2403.14468)!
* 2024 Mar 21: Paper available on [Arxiv](https://arxiv.org/abs/2403.14468).


## ▶️ Quick Start for AnyV2V(i2vgen-xl)
### Environment
Prepare the codebase of the AnyV2V project and Conda environment using the following commands:
```bash
git clone https://github.com/TIGER-AI-Lab/AnyV2V
cd AnyV2V

cd i2vgen-xl
conda env create -f environment.yml
```

### First Frame Image Edit
We provide instructpix2pix port for image editing with instruction prompt.
```shell
usage: edit_image.py [-h] [--model {magicbrush,instructpix2pix}]
                     [--video_path VIDEO_PATH] [--input_dir INPUT_DIR]
                     [--output_dir OUTPUT_DIR] [--prompt PROMPT] [--force_512]
                     [--dict_file DICT_FILE] [--seed SEED]
                     [--negative_prompt NEGATIVE_PROMPT]

Process some images.

optional arguments:
  -h, --help            show this help message and exit
  --model {magicbrush,instructpix2pix}
                        Name of the image editing model
  --video_path VIDEO_PATH
                        Name of the video
  --input_dir INPUT_DIR
                        Directory containing the video
  --output_dir OUTPUT_DIR
                        Directory to save the processed images
  --prompt PROMPT       Instruction prompt for editing
  --force_512           Force resize to 512x512 when feeding into image model
  --dict_file DICT_FILE
                        JSON file containing files, instructions etc.
  --seed SEED           Seed for random number generator
  --negative_prompt NEGATIVE_PROMPT
                        Negative prompt for editing
```

Example usage:
```shell
pip install moviepy
python edit_image.py --video_path "./demo/Man Walking.mp4" --input_dir "./demo" --output_dir "./demo/edited_first_frame" --prompt "turn the man into darth vader"
```

### Inference
Under ```i2vgen-xl/configs/group_ddim_inversion``` and ```i2vgen-xl/configs/group_pnp_edit``` 
1. Modify the ```template.yaml``` files to specify the ``` device```.
2. Modify the ``group_config.json`` files according to the provided examples. Configs in the ``group_config.json`` will overwrite the configs in the ``template.yaml``.
You can set ```active: true``` to enable a example and ```active: false``` to disable a example.

Then you can run the following command to perform inference:
```bash
cd i2vgen-xl/scripts
bash run_group_ddim_inversion.sh
bash run_group_pnp_edit.sh
```
or run the following command using python:
```bash
cd i2vgen-xl/scripts

# First invert the latent of source video
python run_group_ddim_inversion.py \
--template_config "configs/group_ddim_inversion/template.yaml" \
--configs_json "configs/group_ddim_inversion/group_config.json"

# Then run Anyv2v pipeline with the source video latent
python run_group_pnp_edit.py \
--template_config "configs/group_pnp_edit/template.yaml" \
--configs_json "configs/group_pnp_edit/group_config.json"
```

## 📋 TODO
AnyV2V(i2vgen-xl)
- [x] Release the code for AnyV2V(i2vgen-xl)
- [] Release a notebook demo 
- [ ] Release a Gradio demo

AnyV2V(SEINE)
- [ ] Release the code for AnyV2V(SEINE) 

AnyV2V(ConsistI2V)
- [ ] Release the code for AnyV2V(ConsistI2V) 


## 🖊️ Citation

Please kindly cite our paper if you use our code, data, models or results:
```bibtex
@article{ku2024anyv2v,
        title={AnyV2V: A Plug-and-Play Framework For Any Video-to-Video Editing Tasks},
        author={Max Ku and Cong Wei and Weiming Ren and Harry Yang and Wenhu Chen},
        journal={arXiv preprint arXiv:2403.14468},
        year={2024}
        }
```
## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TIGER-AI-Lab/AnyV2V&type=Date)](https://star-history.com/#TIGER-AI-Lab/AnyV2V&Date)
