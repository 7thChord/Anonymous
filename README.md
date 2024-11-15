# Anonymous submission

This repository exhibits the core algorithm described in the paper.

Our codes are based on the following works. Many thanks to the authors.
* Diffusers: https://github.com/huggingface/diffusers
* AnimateDiff: https://github.com/guoyww/AnimateDiff
* Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/main/opensora
  * Our codes are based on v1.1 (released Jun 2024). Compatibility issues may exist for newer versions as OpenSora is under fast update.
* LAMP: https://github.com/RQ-Wu/LAMP

## AnimateDiff version

Dependent models & datasets
* Stable Diffusion v1.5
* Motion module: https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_mm.ckpt
* LAMP dataset: https://github.com/RQ-Wu/LAMP

Core components
* Architecture (including parallelism)
  * SA adapter: `animatediff/models/sa_utils.py`
  * CA adapter: `animatediff/models/ca_utils.py`
  * s-TA adapter: `animatediff/models/motion_module.py`
* Inference-time techniques: `animatediff/pipelines/pipeline_animation.py`

Both training and inference rely on YAML configs.
Please refer to `animatediff/configs/readme.md` about how to prepare a config file. Example YAMLs available under `animatediff/configs`.

## OpenSora version

Dependent models & datasets
* Open-Sora-Plan v1.1: https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main
  * VAE and base model (65x512x512)
* LAMP dataset: https://github.com/RQ-Wu/LAMP

The core components are mostly migrated from AnimateDiff version, with light modifications for compatibility.