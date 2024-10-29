# <img src="https://tiger-ai-lab.github.io/AnyV2V/static/images/icon.png" width="30"/> AnyV2V
[![arXiv](https://img.shields.io/badge/arXiv-2403.14468-b31b1b.svg)](https://arxiv.org/abs/2403.14468)
<a href='https://huggingface.co/papers/2403.14468'><img src='https://img.shields.io/static/v1?label=Paper&message=Huggingface&color=orange'></a> 

<a href='https://huggingface.co/spaces/TIGER-Lab/AnyV2V'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
[![Replicate](https://replicate.com/cjwbw/anyv2v/badge)](https://replicate.com/cjwbw/anyv2v) 

[**🌐 Homepage**](https://tiger-ai-lab.github.io/AnyV2V/)  | [**📖 arXiv**](https://arxiv.org/abs/2403.14468) | [**🤗 HuggingFace Demo**](https://huggingface.co/spaces/TIGER-Lab/AnyV2V) | [**🎬 Replicate Demo**](https://replicate.com/cjwbw/anyv2v) 


[![contributors](https://img.shields.io/github/contributors/TIGER-AI-Lab/AnyV2V)](https://github.com/TIGER-AI-Lab/AnyV2V/graphs/contributors)
[![license](https://img.shields.io/github/license/TIGER-AI-Lab/AnyV2V.svg)](https://github.com/TIGER-AI-Lab/AnyV2V/blob/main/LICENSE)
[![GitHub](https://img.shields.io/github/stars/TIGER-AI-Lab/AnyV2V?style=social)](https://github.com/TIGER-AI-Lab/AnyV2V)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FTIGER-AI-Lab%2FAnyV2V&count_bg=%23C83DB9&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

This repo contains the codebase for our TMLR 2024 paper "[AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks](https://arxiv.org/pdf/2403.14468.pdf)"

## Introduction
AnyV2V is a framework to achieve high appearance and temporal consistency in video editing.
- Perform Video Editing **WITH ONLY SINGLE IMAGE**
  - turning video editing into an image editing problem
  - can seamlessly build on top of image editing methods to perform diverse types of editing
- Training-Free
  - Does not require any training/fine-tuning

<div align="center">
  <img src="assets/AnyV2V-SlidesShow-GIF-1080P-02.gif" alt="AnyV2V" width="70%"/>
</div>


## 📰 News
* 2024 Oct 29: Paper accepted to TMLR 2024.
* 2024 Apr 16: Local Gradio demo now supports edits up to 16 seconds (128 frames).
* 2024 Apr 11: Added local gradio demo for AnyV2V(i2vgen-xl)+[InstantStyle](https://github.com/InstantStyle/InstantStyle).
* 2024 Apr 7: Added sections [the showcases](https://github.com/TIGER-AI-Lab/AnyV2V/issues/6). Share your AnyV2V Edits with us! 
* 2024 Apr 7: We recommend using [InstantStyle](https://github.com/InstantStyle/InstantStyle) with AnyV2V for Video Stylization! Check out [the demo!](https://twitter.com/vinesmsuic/status/1777170927500787782)!
* 2024 Apr 3: [HuggingFace Demo](https://huggingface.co/spaces/TIGER-Lab/AnyV2V) is available!
* 2024 Apr 2: Added local Gradio demo for AnyV2V(i2vgen-xl).
* 2024 Mar 24: Added [Replicate demo](https://replicate.com/cjwbw/anyv2v) for AnyV2V(i2vgen-xl). Thanks [@chenxwh](https://github.com/chenxwh) for the effort!!
* 2024 Mar 22: Code released.
* 2024 Mar 21: Our paper is featured on [Huggingface Daily Papers](https://huggingface.co/papers/2403.14468)!
* 2024 Mar 21: Paper available on [Arxiv](https://arxiv.org/abs/2403.14468). AnyV2V is the first work to leverage I2V models in Video Editing!


## ▶️ Quick Start for AnyV2V(i2vgen-xl)
### Environment
Prepare the codebase of the AnyV2V project and Conda environment using the following commands:
```bash
git clone https://github.com/TIGER-AI-Lab/AnyV2V
cd AnyV2V

cd i2vgen-xl
conda env create -f environment.yml
```

#### 🤗 Local Gradio Demo

AnyV2V+InstructPix2Pix (Prompt-based Editing)
```shell
python gradio_demo.py
```

AnyV2V+InstantStyle Demo (Style Transfer)
```shell
# Download InstantStyle depends
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models
rm -rf IP-Adapter
# Run script
python gradio_demo_style.py
```

#### 📜 Notebook Demo 
We provide a notebook demo ```i2vgen-xl/demo.ipynb``` for AnyV2V(i2vgen-xl).
You can run the notebook to perform Prompt-Based Editing on a single video.
Make sure the environment is set up correctly before running the notebook.

#### To edit multiple demo videos, please refer to the [Video Editing](#Video-Editing) section.

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
or run the following command using Python:
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

## ▶️ Quick Start for AnyV2V(consisti2v)

Please refer to [./consisti2v/README.md](consisti2v/README.md)

## ▶️ Quick Start for AnyV2V(seine)

Please refer to [./seine/README.md](seine/README.md)

## ▶️ Misc

### First Frame Image Edit
We provide the instructpix2pix port for image editing with an instruction prompt.
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

Usage Example:
```shell
python edit_image.py --video_path "./demo/Man Walking.mp4" --input_dir "./demo" --output_dir "./demo/Man Walking/edited_first_frame" --prompt "turn the man into darth vader"
```

You can use other image models for editing, here are some online demo models that you can use:
* [Idenity Manipulation model: InstantID](https://huggingface.co/spaces/InstantX/InstantID)
* [Subject Driven Image editing model: AnyDoor](https://huggingface.co/spaces/xichenhku/AnyDoor-online)
* [Style Transfer model: WISE](https://huggingface.co/spaces/MaxReimann/Whitebox-Style-Transfer-Editing)
* [Style Transfer model: InstantStyle](https://github.com/InstantStyle/InstantStyle)

### Video Preprocess Script

It is possible to edit videos with 16 seconds (128 frames) under an A6000 gpu.
We provide a script to trim and crop video into any dimension and length.

```shell
usage: prepare_video.py [-h] [--input_folder INPUT_FOLDER] [--video_path VIDEO_PATH] [--output_folder OUTPUT_FOLDER]
                        [--clip_duration CLIP_DURATION] [--width WIDTH] [--height HEIGHT] [--start_time START_TIME] [--end_time END_TIME]
                        [--n_frames N_FRAMES] [--center_crop] [--x_offset X_OFFSET] [--y_offset Y_OFFSET] [--longest_to_width]

Crop and resize video segments.

optional arguments:
  -h, --help            show this help message and exit
  --input_folder INPUT_FOLDER
                        Path to the input folder containing video files
  --video_path VIDEO_PATH
                        Path to the input video file
  --output_folder OUTPUT_FOLDER
                        Path to the folder for the output videos
  --clip_duration CLIP_DURATION
                        Duration of the video clips in seconds default=2
  --width WIDTH         Width of the output video (optional) default=512
  --height HEIGHT       Height of the output video (optional) default=512
  --start_time START_TIME
                        Start time for cropping (optional)
  --end_time END_TIME   End time for cropping (optional)
  --n_frames N_FRAMES   Number of frames to extract from each video
  --center_crop         Center crop the video
  --x_offset X_OFFSET   Horizontal offset for center cropping, range -1 to 1 (optional)
  --y_offset Y_OFFSET   Vertical offset for center cropping, range -1 to 1 (optional)
  --longest_to_width    Resize the longest dimension to the specified width
```

Usage Example:
```shell
python prepare_video.py --input_folder src_center_crop/ --output_folder processed --start_time 1 --center_crop --x_offset 0 --y_offset 0
python prepare_video.py --input_folder src_left_crop/ --output_folder processed --start_time 1 --center_crop --x_offset -1 --y_offset 0
python prepare_video.py --input_folder src_right_crop/ --output_folder processed --start_time 1 --center_crop --x_offset 1 --y_offset 0
```

## 📋 TODO
AnyV2V(i2vgen-xl)
- [x] Release the code for AnyV2V(i2vgen-xl)
- [x] Release a notebook demo 
- [x] Release a Gradio demo
- [x] Hosting Gradio demo on HuggingFace Space

AnyV2V(SEINE)
- [x] Release the code for AnyV2V(SEINE) 

AnyV2V(ConsistI2V)
- [x] Release the code for AnyV2V(ConsistI2V) 

Misc
- [x] Helper script to preprocess the source video
- [x] Helper script to obtain edited first frame from the source video

## 🖊️ Citation

Please kindly cite our paper if you use our code, data, models or results:
```bibtex
@article{ku2024anyv2v,
  title={AnyV2V: A Tuning-Free Framework For Any Video-to-Video Editing Tasks},
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

## 💞 Acknowledgements
The code is built upon the below repositories, we thank all the contributors for open-sourcing.
* [diffusers](https://github.com/huggingface/diffusers)
* [ImagenHub](https://github.com/TIGER-AI-Lab/ImagenHub)
* [TokenFlow](https://github.com/omerbt/TokenFlow)
* [i2vgen-xl](https://github.com/ali-vilab/VGen)
* [SEINE](https://github.com/Vchitect/SEINE)
* [ConsistI2V](https://github.com/TIGER-AI-Lab/ConsistI2V)
