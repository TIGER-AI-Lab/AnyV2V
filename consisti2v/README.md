# AnyV2V(_ConsistI2V_)

Our AnyV2V(_ConsistI2V_) is a standalone version.

##  Setup for ConsistI2V

### Prepare Environment
```
conda env create -f environment.yaml
conda activate consisti2v
```

## AnyV2V

**Note:** due to the lower training resolution of ConsistI2V (256x256), it might perform better on 256x256 inputs. We provide configurations for running on both 256x256 and 512x512.

### Run ConsistI2V DDIM Inversion to get the initial latent
Usage Example:
```shell
python run_ddim_inversion.py --config configs/pipeline_256/ddim_inversion_256.yaml video_path=/path/to/your_video.mp4 video_name=your_video
```

Saved latent goes to `./ddim_version` (can be configurated in `./configs/pipeline_256(512)/ddim_inversion_256(512).yaml`).

### Run AnyV2V with ConsistI2V

Your need to prepare your edited image frame first. We provided an image editing script in the root folder of AnyV2V.

Usage Example:
```shell
python run_pnp_edit.py --config configs/pipeline_256/pnp_edit.yaml \
    video_path=/path/to/your_video.mp4 \
    video_name=your_video \
    edited_first_frame_path=/path/to/edited_first_frame.png \
    editing_prompt="<editing_prompt>" \
    ddim_latents_path=/path/to/ddim_latents
```

Saved video goes to `./anyv2v_results` (can be configurated in `./configs/pipeline_256(512)/pnp_edit.yaml`).
