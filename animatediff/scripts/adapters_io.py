import argparse
from omegaconf import OmegaConf
from safetensors import safe_open
import torch
from typing import Dict, Optional, Tuple

def separate_pretrained_checkpoint(
    output_dir: str,
    pretrained_model_path: str,
    motion_module_path: str,
    use_ip_adapter: bool,
    ip_ckpt: str,
    image_encoder_path: str,
    validation_data: Dict,
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    noise_scheduler_kwargs = None,
    trainable_modules = ["i2v_adapter.to_out", "to_q_lora", "to_k_lora", "to_v_lora", "to_out_lora"],
    i2v_weight_path: str = "animatediff-i2v-nv1-motionlora-i2v-weights.pt",
    lora_weight_path: str = "animatediff-i2v-nv1-motionlora-lora-weights.pt",
    ca_lora_weight_path: str = "animatediff-i2v-nv1-ca-lora-weights.pt",
    combined_path: str = "combined.pt"
):
    combined_dict = {}
    with safe_open(unet_checkpoint_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if any(module in k for module in trainable_modules):
                combined_dict[k] = f.get_tensor(k)

    torch.save(combined_dict, combined_path)
    print("Finished Checkpoint Split")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    separate_pretrained_checkpoint(**config)
