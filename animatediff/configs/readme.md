- [Config README](#config-readme)
  - [Header - Base checkpoints](#header---base-checkpoints)
  - [Header - Load the adapters](#header---load-the-adapters)
    - [`load_from_adapter_ckpt`](#load_from_adapter_ckpt)
    - [Use multiple adapters](#use-multiple-adapters)
  - [Header - Learning textual embeddings](#header---learning-textual-embeddings)
  - [Header - Other values for training](#header---other-values-for-training)
  - [Section `unet_additional_kwargs`: the UNet architecture](#section-unet_additional_kwargs-the-unet-architecture)
    - [Motion module](#motion-module)
    - [CA-LoRA](#ca-lora)
    - [Spatiotemporal $Q$ in CA](#spatiotemporal-q-in-ca)
    - [Orthogonal Adapters (OA)](#orthogonal-adapters-oa)
    - [Multi-adapter setting](#multi-adapter-setting)
  - [Section `validation_data`](#section-validation_data)
    - [Input](#input)
    - [Output](#output)
    - [Sampling process](#sampling-process)
    - [Initial noise](#initial-noise)
    - [Multi-adapter](#multi-adapter)
    - [Learnable text embedding](#learnable-text-embedding)
    - [Visualizing CA maps](#visualizing-ca-maps)
  - [Section `noise_scheduler_kwargs`](#section-noise_scheduler_kwargs)


# Config README
The config file structure is largely based on AnimateDiff, but we make some differences.

## Header - Base checkpoints
The first part of the config file specifies the skeleton of the video DM.
* Essentially it is a T2I model (e.g. SD) + motion modules aka temporal AttBlocks (e.g. AnimateDiff) + optionally IP-Adapter.
  
| Name        | Notes           |
| ------------- |-------------| 
| `output_dir`    | Where the inference results will be stored |
| `pretrained_model_path`     | Path to the pretrained T2I model   |
| `motion_module_path` | Path to the pretrained motion module (AnimateDiff) |
| `use_ip_adapter` | Whether to include IP-Adapter (default=False) |
| `ip_ckpt`     | (if `use_ip_adapter==True`) Path to the pretrained IP-Adapter   |
| `image_encoder_path` | (if `use_ip_adapter==True`) Path to the pretrained image encoder (in IPAdapter) |
| `textemb_checkpoint_path` | Optional. The path to the textual embedding for [customized textual embedding case](#learnable-text-embedding). |
| `global_seed`: int | The global seed for random number generation. |

## Header - Load the adapters 

### `load_from_adapter_ckpt`
* `False` (default) \
Traditionally, all weights including those not fine-tuned are saved into one `.safetensors` file per training task. In this case, users need to specify
  * `unet_checkpoint_path`
* `True` \
We also offer the option to save/load fine-tuned weights only, in which case a task produces at most 3 checkpoints. In this case, users need to specify:
  * `adapter_i2v_path`: The I2V-Adapter weights at SA.
  * `adapter_lora_path`: The MotionLoRA at temporal layers.
  * `ca_lora_path`: The CA-LoRA at CA.
  * Any attribute can be left as `None` if not involved in the training task.

### Use multiple adapters

NOTE: inference only - not supported by training.

In this case, all the above attributes should be a list. Example:

```yaml
# Option 1: load multiple full ckpts; each ckpt contains a full UNet including frozen weights
load_from_adapter_ckpt : False
unet_checkpoint_path: 
  - "checkpoints/model1.safetensors"
  - "checkpoints/model2.safetensors"

# Option 2: load fine-tuned weights only
# NOTE: all list lengths should match.
load_from_adapter_ckpt : True
adapter_i2v_path : 
  - adapters/Adapter1-i2v-weights.pt # inference: can be multiple; training: only one
  - adapters/Adapter2-i2v-weights.pt
adapter_lora_path : 
  - adapters/Adapter1-lora-weights.pt
  - adapters/Adapter2-lora-weights.pt
ca_lora_path : 
  - adapters/Adapter1-ca-lora-weights.pt
  - adapters/Adapter2-ca-lora-weights.pt
```

## Header - Learning textual embeddings

Below arguments work in training only.

| Name        | Notes           |
| ------------- |-------------| 
|`word_to_learn`: str| If set, the training task also learn a new word embedding for the specified word. Don't include this if the task won't learn word embeddings. |
|`initializer_token`: str| If `word_to_learn` is set, the trainer initializes the word embedding of `word_to_learn` as this. |

* Example: `word_to_learn` = "clouds_new", `initializer_token` = "clouds"

## Header - Other values for training

Below arguments work in training only.

| Name        | Notes           |
| ------------- |-------------| 
| `split_ckpt`: bool | `True` to store only the fine-tuned weights, `False` to store all UNet weights in a `.safetensors` file. Default = `False`. |
|`trainable_modules`: list[str] | Specify which modules to attach LoRA. |

## Section `unet_additional_kwargs`: the UNet architecture

This section specifies the DM's architecture.
The loaded checkpoints must follow the definition here or misalignment errors may occur.

NOTE: some attributes are omitted - please just don't modify them.

### Motion module

| Name        | Notes           |
| ------------- |-------------| 
|`use_motion_module`: bool    | Whether to use motion modules (temporal AttBlocks) |
|`use_motion_embed`: bool | Under `motion_module_kwargs` section. If True, then learn a motion embedding following [Motion-Inv]. |

### CA-LoRA

| Name        | Notes           |
| ------------- |-------------| 
|`use_ca_lora`: bool    | Whether to use CA-LoRA |
|`ca_lora_rank`: int / list[int]    | Rank of CA-LoRA(s). For single adapter, it's an *int* (e.g. `32`); for multi-adapter, it's a list of ranks for each adapter (e.g. `[32,32]`)|
|`ca_lora_scale`: float / list[float]    | `network_alpha` value of CA-LoRA(s). See lora.py for details. Data type works the same as `ca_lora_rank`|
|`use_q_lora`: bool  | if True, attach LoRA to W_Q. Default = `False` (only attach LoRAs to W_K,V,O).|

### Spatiotemporal $Q$ in CA

| Name        | Notes           |
| ------------- |-------------| 
|`q_downsample`: bool    | Whether to use spatiotemporal $Q$ in CA |
|`q_downsample_ratio`: int    | The stride of spatial downsampling in $Q$. Default = 4.|
|`ca_pe_mode`: None ("null") / "naive" / "temporal" / "temporal_sine" / "rope_1d" / "rope_3d" / "ropeQ_3d" / "ropeQK_1d" / "ropeQKV_1d"  | The positional embedding setting for the query tensor in CA. Default = None. |

### Orthogonal Adapters (OA)

Leave this part blank if not used (remove the related attributes at all).

| Name        | Notes           |
| ------------- |-------------| 
|`oa_bin_id`: int    | The bases bin ID. An adapter w.r.t. a particular movement should have a unique ID. |
|`oa_bases_path`: str    | The path that stores the pre-computed LoRA bases. Users are assumed to run `scripts/lora_basis_lib/create.py` to generate the bases first.|

### Multi-adapter setting
| Name        | Notes           |
| ------------- |-------------| 
| `adapter_weights`: list[float]   | FOR INFERENCE ONLY. Weight of each adapter during inference. Default = `[1]`. For two-adapter setting, `[0.5,0.5]` can be a good start.|
| `parallel_mode`: "weights" / "residual" | FOR INFERENCE ONLY. Determines how multiple adapters are parallelized. Default = "residual". "weights" follows original LoRA practice by adding all LoRAs to the pre-trained weights.|


## Section `validation_data`

The setting for validation phase (training) and inference (inference). 

### Input

| Name        | Notes           |
| ------------- |-------------| 
|`image_path`: str| The path to the input images (assuming all input images are stored in the same folder)|
|`prompt_path`: list[str] | List of image file names. The program will run through all files. |
|`prompts`: list[str] / "all" / "newlamp"| A list whose length == `prompt_path`, specifying the text prompt per input image. |
|`batch_prompt`: None(default) / str / list[str]| Use this if the prompt is consistent for all input images. *In multi-adapter case,* `batch_prompt` will override `prompts` and must be a list of str, specifying the prompt injected to each adapter respectively (e.g. `["waterfall", "cloud"]`). |

### Output
| Name        | Notes           |
| ------------- |-------------|
|`video_length`: int| As is. Default = 16.|
|`width`, `height`: int| As is. Default = 512, 320.|

### Sampling process
| Name        | Notes           |
| ------------- |-------------|
|`num_inference_steps`: int| Time step # for the sampler. Default = 50.|
|`guidance_scale`: float | CFG guidance scale|
|`use_adain_norm`: "off" / "on" / "last" | AdaIN behaviour in the sampling process. "off": no AdaIN. "on": perform AdaIN after each timestep (by [LAMP]); perform AdaIN only after the last timestep. Default = "off".|

### Initial noise
| Name        | Notes           |
| ------------- |-------------|
|`shared_noise_ratio`: float | Ratio of shared noise across all frames in $z_T$. [LAMP] uses 0.2. We find 0.05 is good for our setting. Default = 0. |
|`first_frame_cond_type`: "lamp" / "cinemo" | "lamp": Initial noise is pure noise with optional shared-noise mechanism. "cinemo": Initial noise is a noisy signal by forward diffusion process. Must be "cinemo" if the user would use DCTInit initalization. Default = "lamp". |
|`use_dct_init`: bool| If `True` and "cinemo" flag, apply DCTInit - integrate low-frequency component of the latent image and high-frequency component of random noise as the initial noise. |
|`dct_cutoff_ratio`: float| (for DCTInit case only) The cutoff ratio in [0,1] for DCTInit. Default = 0.23. |
|`dct_cutoff_shape`: 'rect' / 'tri'| (for DCTInit case only) How DCT frequency domain is divided at top left, in rectangle or triangle shape. default = "rect". |
|`cil_ratio`: float| Default = 1. Otherwise, all the timesteps in the schedule will be multiplied by it following [CIL]. |

### Multi-adapter
| Name        | Notes           |
| ------------- |-------------|
|`multi_ca`: bool| Set as `True` to switch on multi-adapter setting. Default = `False`. |
|`batch_prompt`: list[str]| See above.|
|`text_emb_dir`: str| (for multi-adapter case only) If given, the program will obtain the text embeddings offline. Specifically, collect all prompts from `batch_prompt`, and then search for corresponding embedding file in this folder; for example, if a prompt is "waterfall", then the code will try to load `text_emb_dir/waterfall.pt`. Users must generate the embedding in advance to enable this feature. Default = "text_embs". |

### Learnable text embedding

| Name        | Notes           |
| ------------- |-------------|
|`word_to_replace`: str / list[str] | If set, the program will replace the embedding of this word with a learned embedding. Should be a list under multi-adapter setting. |

* The embedding is specified by `textemb_checkpoint_path` in the header. Otherwise the code assumes
  * UNet ckpt at `unet_checkpoint_path`/checkpoints/model.safetensors
  * Text-emb at `unet_checkpoint_path`/learned_embeds.safetensors

### Visualizing CA maps

| Name        | Notes           |
| ------------- |-------------|
|`CA_Map_visualization_resolution`: list[str] / "all" / `None` | If set, output the CA maps per frame. Can be any combination of "low", "middle" and "high", or just "all" to output them all. Recommended to perform visualization on checkpoints without `q_downsample_ratio > 1` only. Not supporting `multi_ca==True` case yet. Default is `None`.|

## Section `noise_scheduler_kwargs`
The noise schedule setting. In default it follows SD's DDPM schedule.