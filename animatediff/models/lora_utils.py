import os
import torch
from tqdm import tqdm
from torch import nn


def gen_orthonormal_basis_set(dim):
    A = torch.randn(dim, dim)
    U,S,V = torch.linalg.svd(A)
    bases = U@V
    return bases # orth[:,i] is a basis


def presave_basis_set(dim, path):
    bases = gen_orthonormal_basis_set(dim)
    torch.save(bases, path)


def replace_oa(dict_lora: dict, basis_set_dir: str, bin_id=None,
               ignore_ca_kv=False):
    '''
    Replace all LoRAs (untrained) with OAs by freezing "up" (B) sub-layers.
    CALL THIS BEFORE TRAINING!
    * dict_lora: the LoRA set. Should be given by `UNet.lora_layers`.
    * basis_set_dir: the pre-generated orthonormal bases (320/640/1280-d)
    * output_picked_ids_path: if specified, store `picked_ids` here.
    * ignore_ca_kv: if True, don't apply OA on CA's to_k_lora and to_v_lora.
    -------
    v2: basis_set_dir contains {key}.pt files. For each layer `key`, draw basis from the corresponding .pt file.
    * bin_id: the bases are divided into `bins`, where each bin contains `rank` bases.
        * if None, then pick bases randomly.
        * if bin_id exceeds # of bins, take the modulo instead.
    Ignore the conflicts before multiple adapters.
    '''
    for k, v in tqdm(dict_lora.items()):
        if ignore_ca_kv:
            if "attn2.processor.to_k_lora" in k or "attn2.processor.to_v_lora" in k:
                continue
        resolution = v.in_features # assume in_features == out_features
        rank = v.rank
        # load bases
        path = os.path.join(basis_set_dir, f"{k}.pt")
        bases = torch.load(path)
        n_bases = bases.shape[1]
        n_bins = n_bases // rank
        # draw bases
        if bin_id is None:
            basis_ids = torch.randint(0, resolution, (rank,))
        else:
            bin_id = bin_id % n_bins
            basis_ids = list(range(bin_id*rank, (bin_id+1)*rank))

        bases = bases[:, basis_ids]
        
        with torch.no_grad():
            v.up.weight.copy_(bases)

        v.up.weight.requires_grad = False # freeze B
        nn.init.zeros_(v.down.weight) # init A as zero

    print(f"[lora_utils]Voila. LoRA_up layers now replaced by OA. rank = {rank}.")


if __name__=='__main__':
    # an example to use OA
    from omegaconf import OmegaConf
    from animatediff.models.unet import UNet3DConditionModel

    config = OmegaConf.load("configs/inference/inference-aniamtediff-i2v.yaml")
    unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_model_path,
        config.motion_module_path,
            subfolder="unet", 
        unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs)
    )

    ORTH_BASES_DIR = "configs/lora_bases"
    replace_oa(unet.lora_layers(), ORTH_BASES_DIR, bin_id=0)