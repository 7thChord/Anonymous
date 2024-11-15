import abc
import torch
import numpy as np
import os
import math
from PIL import Image
from einops import rearrange
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return 0


    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def show_result(self):
        return {"attention_store": self.attention_store, "step_store":self.step_store}


    def store(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        
        if True: 
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):

        self.cur_step += 1
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store() 

    def get_average_attention(self): #
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


def show_cross_attention(attention_store: AttentionStore, width = 16, height = 10, from_where = ["up", "down"], q_downsample = -1, output_path = ""):
    
    is_cross = True
    images = []
    out = []
    attention_maps = attention_store.get_average_attention() 

    res = width * height
    if q_downsample!= -1:
        res = res * 16

    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[2] == res:
                cross_maps = item.reshape(-1, 8, 16, height, width, item.shape[-1])
                cross_maps = rearrange(cross_maps, "b c f h w d -> (b f) c h w d")

                out.append(cross_maps)
  
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if len(out) == 0:
        print("No map exist at given resolution")
        return

    for i in range(out[0].shape[0]):
        out_i = []
        for c_map in out:
            out_i.append(c_map[i])

        out_i = torch.cat(out_i, dim=0)
        out_i = out_i.sum(0) / out_i.shape[0]

        display_image(out_i[:,:,1], f"{output_path}/{i}.jpg")

def display_image(image, output_path):
    image = image.cpu()
    image = 255 * image / image.max()
    image = image.unsqueeze(-1).expand(*image.shape, 3)
    image = image.numpy().astype(np.uint8)
    image = np.array(Image.fromarray(image).resize((512, 320)))
    pil_img = Image.fromarray(image)
    pil_img.save(output_path)
