import os, io, csv, math, random
import json
import numpy as np
from einops import rearrange
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from diffusers.utils import logging

logger = logging.get_logger(__name__)

class WebVid10M(Dataset):
    def __init__(
            self,
            json_path, video_folder=None,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
            **kwargs,
        ):
        logger.info(f"loading annotations from {json_path} ...")
        with open(json_path, 'rb') as json_file:
            json_list = list(json_file)
        self.dataset = [json.loads(json_str) for json_str in json_list]
        self.length = len(self.dataset)
        logger.info(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride if isinstance(sample_stride, int) else tuple(sample_stride)
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0], antialias=None),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_relative_path, name = video_dict['file'], video_dict['text']
        
        if self.video_folder is not None:
            if video_relative_path[0] == '/':
                video_dir = os.path.join(self.video_folder, os.path.basename(video_relative_path))
            else:
                video_dir = os.path.join(self.video_folder, video_relative_path)
        else:
            video_dir = video_relative_path
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        
        if not self.is_image:
            if isinstance(self.sample_stride, int):
                stride = self.sample_stride
            elif isinstance(self.sample_stride, tuple):
                stride = random.randint(self.sample_stride[0], self.sample_stride[1])
            clip_length = min(video_length, (self.sample_n_frames - 1) * stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            frame_difference = random.randint(2, self.sample_n_frames)
            clip_length = min(video_length, (frame_difference - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = [start_idx, start_idx + clip_length - 1]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader
        
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample


class Pexels(Dataset):
    def __init__(
            self,
            json_path, caption_json_path, video_folder=None,
            sample_size=256, sample_duration=1, sample_fps=8,
            is_image=False,
            **kwargs,
        ):
        logger.info(f"loading captions from {caption_json_path} ...")
        with open(caption_json_path, 'rb') as caption_json_file:
            caption_json_list = list(caption_json_file)
        self.caption_dict = {json.loads(json_str)['id']: json.loads(json_str)['text'] for json_str in caption_json_list}
        
        logger.info(f"loading annotations from {json_path} ...")
        with open(json_path, 'rb') as json_file:
            json_list = list(json_file)
        dataset = [json.loads(json_str) for json_str in json_list]

        self.dataset = []
        for data in dataset:
            data['text'] = self.caption_dict[data['id']]
            if data['height'] / data['width'] < 0.625:
                self.dataset.append(data)
        self.length = len(self.dataset)
        logger.info(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_duration = sample_duration
        self.sample_fps      = sample_fps
        self.sample_n_frames = sample_duration * sample_fps
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0], antialias=None),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_relative_path, name = video_dict['file'], video_dict['text']
        fps = video_dict['fps']
        
        if self.video_folder is not None:
            if video_relative_path[0] == '/':
                video_dir = os.path.join(self.video_folder, os.path.basename(video_relative_path))
            else:
                video_dir = os.path.join(self.video_folder, video_relative_path)
        else:
            video_dir = video_relative_path
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, math.ceil(fps * self.sample_duration))
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            frame_difference = random.randint(2, self.sample_n_frames)
            sample_stride = math.ceil((fps * self.sample_duration) / (self.sample_n_frames - 1) - 1)
            clip_length = min(video_length, (frame_difference - 1) * sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = [start_idx, start_idx + clip_length - 1]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader
        
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample


class JointDataset(Dataset):
    def __init__(
            self,
            webvid_config, pexels_config,
            sample_size=256,
            sample_duration=None, sample_fps=None, sample_stride=None, sample_n_frames=None,
            is_image=False,
            **kwargs,
        ):
        assert (sample_duration is None and sample_fps is None) or (sample_duration is not None and sample_fps is not None), "sample_duration and sample_fps should be both None or not None"
        if sample_duration is not None and sample_fps is not None:
            assert sample_stride is None, "when sample_duration and sample_fps are not None, sample_stride should be None"
        if sample_stride is not None:
            assert sample_fps is None and sample_duration is None, "when sample_stride is not None, sample_duration and sample_fps should be both None"

        self.dataset = []

        if pexels_config.enable:
            logger.info(f"loading pexels dataset")
            logger.info(f"loading captions from {pexels_config.caption_json_path} ...")
            with open(pexels_config.caption_json_path, 'rb') as caption_json_file:
                caption_json_list = list(caption_json_file)
            self.caption_dict = {json.loads(json_str)['id']: json.loads(json_str)['text'] for json_str in caption_json_list}
            
            logger.info(f"loading annotations from {pexels_config.json_path} ...")
            with open(pexels_config.json_path, 'rb') as json_file:
                json_list = list(json_file)
            dataset = [json.loads(json_str) for json_str in json_list]

            for data in dataset:
                data['text'] = self.caption_dict[data['id']]
                data['dataset'] = 'pexels'
                if data['height'] / data['width'] < 0.625:
                    self.dataset.append(data)
        
        if webvid_config.enable:
            logger.info(f"loading webvid dataset")
            logger.info(f"loading annotations from {webvid_config.json_path} ...")
            with open(webvid_config.json_path, 'rb') as json_file:
                json_list = list(json_file)
            dataset = [json.loads(json_str) for json_str in json_list]
            for data in dataset:
                data['dataset'] = 'webvid'
            self.dataset.extend(dataset)

        self.length = len(self.dataset)
        logger.info(f"data scale: {self.length}")

        self.pexels_folder   = pexels_config.video_folder
        self.webvid_folder   = webvid_config.video_folder
        self.sample_duration = sample_duration
        self.sample_fps      = sample_fps
        self.sample_n_frames = sample_duration * sample_fps if sample_n_frames is None else sample_n_frames
        self.sample_stride   = sample_stride if (sample_stride is None) or (sample_stride is not None and isinstance(sample_stride, int)) else tuple(sample_stride)
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0], antialias=None),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_relative_path, name = video_dict['file'], video_dict['text']

        if video_dict['dataset'] == 'pexels':
            video_folder = self.pexels_folder
        elif video_dict['dataset'] == 'webvid':
            video_folder = self.webvid_folder
        else:
            raise NotImplementedError
        
        if video_folder is not None:
            if video_relative_path[0] == '/':
                video_dir = os.path.join(video_folder, os.path.basename(video_relative_path))
            else:
                video_dir = os.path.join(video_folder, video_relative_path)
        else:
            video_dir = video_relative_path
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        
        stride = None
        if not self.is_image:
            if self.sample_duration is not None:
                fps = video_dict['fps']
                clip_length = min(video_length, math.ceil(fps * self.sample_duration))
            elif self.sample_stride is not None:
                if isinstance(self.sample_stride, int):
                    stride = self.sample_stride
                elif isinstance(self.sample_stride, tuple):
                    stride = random.randint(self.sample_stride[0], self.sample_stride[1])
                clip_length = min(video_length, (self.sample_n_frames - 1) * stride + 1)

            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)

        else:
            frame_difference = random.randint(2, self.sample_n_frames)
            if self.sample_duration is not None:
                fps = video_dict['fps']
                sample_stride = math.ceil((fps * self.sample_duration) / (self.sample_n_frames - 1) - 1)
            elif self.sample_stride is not None:
                sample_stride = self.sample_stride
            
            clip_length = min(video_length, (frame_difference - 1) * sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = [start_idx, start_idx + clip_length - 1]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader
        
        return pixel_values, name, stride

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name, stride = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name, stride=stride)
        return sample
