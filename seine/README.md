# AnyV2V(_SEINE_)

Our AnyV2V(_SEINE_) is a standalone version.

##  Setup for SEINE

### Prepare Environment
```
conda create -n seine python==3.9.16
conda activate seine
pip install -r requirement.txt
```

### Download SEINE model and T2I base model

SEINE model is based on Stable diffusion v1.4, you may download [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) to the director of ``` pretrained ```
.
Download SEINE model checkpoint (from [google drive](https://drive.google.com/drive/folders/1cWfeDzKJhpb0m6HA5DoMOH0_ItuUY95b?usp=sharing) or [hugging face](https://huggingface.co/xinyuanc91/SEINE/tree/main)) and save to the directory of ```pretrained```


Now under `./pretrained`, you should be able to see the following:
```
├── pretrained
│   ├── seine.pt
│   ├── stable-diffusion-v1-4
│   │   ├── ...
└── └── ├── ...
        ├── ...
```

## AnyV2V

### Configure paths for SEINE models

Edit the model paths in both yaml files:
* `./configs/ddim_inversion.yaml`
* `./configs/pnp_edit.yaml`

```yaml
# Model
model_name: "seine"
sd_path: "<your_path>/stable-diffusion-v1-4"
ckpt_path: "<your_path>/SEINE/seine.pt"
model_key: "<your_path>/stable-diffusion-v1-4"
```

Theortically, `<your_path>` should equal to `./pretrained`.


### Run SEINE DDIM Inversion to get the initial latent
```shell
usage: run_ddim_inversion.py [-h] [--config CONFIG] [--video_path VIDEO_PATH] [--gpu GPU]
                             [--width WIDTH] [--height HEIGHT]

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --video_path VIDEO_PATH
                        Path to the video to invert.
  --gpu GPU             GPU number to use.
  --width WIDTH
  --height HEIGHT
```

Usage Example:
```shell
python run_ddim_inversion.py --gpu 0 --video_path "../demo/Man Walking.mp4" --width 512 --height 512
```

Saved latent goes to `./ddim_version` (can be configurated in `./configs/ddim_inversion.yaml`).

### Run AnyV2V with SEINE

Your need to prepare your edited image frame first. We provided an image editing script in the root folder of AnyV2V.

```shell
python run_pnp_edit.py --config ./configs/pnp_edit.yaml \
    src_video_path="your_video.mp4" \
    edited_first_frame_path="your edited first frame image.png" \
    prompt="your prompt" \
    device="cuda:0"
```

Usage Example:
```shell
python run_pnp_edit.py --config ./configs/pnp_edit.yaml \
    src_video_path="../demo/Man Walking.mp4" \
    edited_first_frame_path="../demo/Man Walking/edited_first_frame/turn the man into darth vader.png" \
    prompt="Darth Vader Walking"
```

Saved video goes to `./anyv2v_results` (can be configurated in `./configs/pnp_edit.yaml`).
