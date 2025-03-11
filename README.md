<div align="center">
  <img src="./.asset/favicon.png" width="30%">
</div>

# Manga Animation Xtend

**Stable Video Diffusion for Manga Animation 🚀**

## :bulb: Highlight

- **Finetuning Stable Video Diffusion for Manga Animation.** See [Part 1](#part-1-training).


## Part 1: Training

### Model Configuration
```python
size=(512, 320), motion_bucket_id=127, fps=7, noise_aug_strength=0.00
generator=torch.manual_seed(111)
```

### Data Preparation
Our framework supports any manga or illustrated sequence as input. Ensure your dataset is structured as follows:
```bash
self.base_folder
    ├── manga_sequence1
    │   ├── frame_0001.png
    │   ├── frame_0002.png
    │   ...
    ├── manga_sequence2
    │   ├── frame_0001.png
        ├── ...
```
### Training Configuration
This configuration is a reference setup, where all `unet` parameters are trainable, and we use a learning rate of `1e-5`.
```bash
accelerate launch train_svd_lora.py \
    --pretrained_model_name_or_path=/path/to/weight \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --width=512 \
    --height=320 \
    --checkpointing_steps=1000 --checkpoints_total_limit=1 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200
```



## :label: Installation
To install on **Linux**, simply run:
```bash
sh linux_setup.sh
```

## :label: TODO List

- [ ] More conditioning for UNet (WIP)


## :hearts: Acknowledgement

This project builds upon [Diffusers](https://github.com/huggingface/diffusers) and [Stability AI](https://github.com/Stability-AI/generative-models). Special thanks to their work!

We also acknowledge the research contributions from [FrameWarpNet](https://arxiv.org/abs/2402.01566) and [StyleGAN](https://github.com/NVlabs/stylegan3).


