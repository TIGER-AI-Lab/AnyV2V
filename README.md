# <img src="https://tiger-ai-lab.github.io/AnyV2V/static/images/icon.png" width="30"/> AnyV2V
[![arXiv](https://img.shields.io/badge/arXiv-2403.14468-b31b1b.svg)](https://arxiv.org/abs/2403.14468)
[![contributors](https://img.shields.io/github/contributors/TIGER-AI-Lab/AnyV2V)](https://github.com/TIGER-AI-Lab/AnyV2V/graphs/contributors)
[![open issues](https://isitmaintained.com/badge/open/TIGER-AI-Lab/AnyV2V.svg)](https://github.com/TIGER-AI-Lab/AnyV2V/issues)
[![pull requests](https://img.shields.io/github/issues-pr/TIGER-AI-Lab/AnyV2V?color=0088ff)](https://github.com/TIGER-AI-Lab/AnyV2V/pulls)
[![license](https://img.shields.io/github/license/TIGER-AI-Lab/AnyV2V.svg)](https://github.com/TIGER-AI-Lab/AnyV2V/blob/main/LICENSE)

[**🌐 Homepage**](https://tiger-ai-lab.github.io/AnyV2V/)  | [**🤗 HuggingFace Paper**](https://huggingface.co/papers/2403.14468) | [**📖 arXiv**](https://arxiv.org/abs/2403.14468) | [**GitHub**](https://github.com/TIGER-AI-Lab/AnyV2V)


This repo contains the codebase for the paper "[AnyV2V: A Plug-and-Play Framework For Any Video-to-Video Editing Tasks](https://arxiv.org/pdf/2403.14468.pdf)"

<div align="center">
  <img src="assets/AnyV2V-SlidesShow-GIF-1080P-02.gif" alt="AnyV2V" width="70%"/>
</div>

## Introduction
AnyV2V is a tuning-free framework to achieve high appearance and temporal consistency in video editing.
- can seamlessly build on top of advanced image editing methods to perform diverse types of editing
- robust performance on the four tasks:
  - prompt-based editing
  - reference-based style transfer
  - subject-driven editing
  - identity manipulation


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

#### 📜 Notebook Demo 
We provide a notebook demo ```i2vgen-xl/demo.ipynb``` for AnyV2V(i2vgen-xl).
You can run the notebook to perform a Prompt-Based Editing on a single video.
Make sure the environment is set up correctly before running the notebook.

#### To edit multiple demo videos, please refer to the [Video Editing](#Video-Editing) section.

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
python edit_image.py --video_path "./demo/Man Walking.mp4" --input_dir "./demo" --output_dir "./demo/Man Walking/edited_first_frame" --prompt "turn the man into darth vader"
```

You can use other image models for editing, here are some online demo models that you can use:
* [Idenity Manipulation model: InstantID](https://huggingface.co/spaces/InstantX/InstantID)
* [Subject Driven Image editing model: AnyDoor](https://huggingface.co/spaces/xichenhku/AnyDoor-online)
* [Style Transfer model: WISE](https://huggingface.co/spaces/MaxReimann/Whitebox-Style-Transfer-Editing)

### Video Editing
We provide demo source videos and edited images in the ```demo``` folder. 
Below are the instructions for performing video editing on the provided source videos. 
Navigate to ```i2vgen-xl/configs/group_ddim_inversion``` and ```i2vgen-xl/configs/group_pnp_edit```:
1. Modify the ```template.yaml``` files to specify the ```device```.
2. Modify the ```group_config.json``` files according to the provided examples. The configurations in ```group_config.json``` will override the configurations in ```template.yaml```.
To enable an example, set ```active: true```; to disable it, set ```active: false```.

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

#### To edit your own source videos, follow the steps outlined below:
1. Prepare the source video ```Your-Video.mp4```in the ```demo``` folder.
2. Create two new folders ```demo/Your-Video-Name``` and  ```demo/Your-Video-Name/edited_first_frame```.
3. Run the following command to perform first frame image editing:
```bash
python edit_image.py --video_path "./demo/Your-Video.mp4" --input_dir "./demo" --output_dir "./demo/Your-Video-Name/edited_first_frame" --prompt "Your prompt"
```
You can also use any other image editing method, such as InstantID, AnyDoor, or WISE, to edit the first frame.
Please put the edited first frame images in the ```demo/Your-Video-Name/edited_first_frame``` folder.

4. Add an entry to the ```group_config.json``` files located in ```i2vgen-xl/configs/group_ddim_inversion``` and ```i2vgen-xl/configs/group_pnp_edit``` directories for your video, following the provided examples. 
5. Run the inference command:
```bash
cd i2vgen-xl/scripts
bash run_group_ddim_inversion.sh
bash run_group_pnp_edit.sh
```


## 📋 TODO
AnyV2V(i2vgen-xl)
- [x] Release the code for AnyV2V(i2vgen-xl)
- [x] Release a notebook demo 
- [ ] Release scripts for multiple image editing
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
  author={Ku, Max and Wei, Cong and Ren, Weiming and Yang, Harry and Chen, Wenhu},
  journal={arXiv preprint arXiv:2403.14468},
  year={2024}
}
```

## 🎫 License

This project is released under the [the MIT License](LICENSE).
However, our code is based on some projects that might used another license:

* [i2vgen-xl](https://github.com/ali-vilab/VGen): Missing License
* [SEINE](https://github.com/Vchitect/SEINE): Apache-2.0
* [ConsistI2V](https://github.com/TIGER-AI-Lab/ConsistI2V): MIT License

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TIGER-AI-Lab/AnyV2V&type=Date)](https://star-history.com/#TIGER-AI-Lab/AnyV2V&Date)

## 📞 Contact Authors
Max Ku [@vinemsuic](https://github.com/vinesmsuic), m3ku@uwaterloo.ca
<br>
Cong Wei [@lim142857](https://github.com/lim142857), c58wei@uwaterloo.ca
<br>
Weiming Ren [@wren93](https://github.com/wren93), w2ren@uwaterloo.ca
<br>
