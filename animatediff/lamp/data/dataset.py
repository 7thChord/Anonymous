import decord
decord.bridge.set_bridge('torch')
import os
import cv2
import numpy as np

from torch.utils.data import Dataset
from einops import rearrange
import random
from torchvision.transforms import Resize
import torch
import json

class LAMPDataset(Dataset):
    def __init__(
            self,
            video_root: str,
            prompt: str = None,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_root = video_root
        self.video_path = []
        self.prompt = []
        # for video dataset with multiple prompts instead of one single prompt
        if os.path.isfile(video_root) and video_root.endswith('.json'):
            with open(video_root, 'r') as f:
                data = json.load(f)
                for item in data:
                    self.video_path.append(item['vid'])
                    self.prompt.append(item['cap'])
        elif os.path.isdir(video_root):
            if prompt is None:
                print("Error: Prompt is needed!")
                exit()
            for video_name in os.listdir(video_root):
                self.video_path.append(os.path.join(self.video_root, video_name))
                # if video_root.endswith('/'):
                #     prompt = video_root.split('/')[-2].replace('_', ' ')
                # else:
                #     prompt = video_root.split('/')[-1].replace('_', ' ')
                self.prompt.append(prompt)
        else:
            if prompt is None:
                print("Error: Prompt is needed!")
                exit()
            self.video_path.append(video_root)
            # self.prompt.append(video_root.split('/')[-1].replace('_', ' ').replace('.mp4', ''))
            self.prompt.append(prompt)

        self.prompt_ids = []

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, index):
        vr = decord.VideoReader(self.video_path[index], width=self.width, height=self.height)

        start_idx = random.randint(0, len(vr)-self.n_sample_frames*self.sample_frame_rate-1)
        sample_index = list(range(start_idx, len(vr), self.sample_frame_rate))[:self.n_sample_frames]

        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")
        if random.uniform(0, 1) > 0.5:
            video = torch.flip(video, dims=[3])
        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids[index]
        }

        return example
