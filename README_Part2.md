# README_Part2: SDXL LoRA Fine-tuning & Evaluation Framework (Custom Edition)

This document outlines the custom modules and workflows added to the original Stability AI `generative-models` (SGM) repository. These enhancements transform the research codebase into a production-ready LoRA fine-tuning workstation.

---

## 🚀 New Customized Features

We have injected several high-level modules to handle the full lifecycle of a fine-tuning project, from raw images to visual evaluation.

### 1. Native LoRA Injection Engine (`sgm/modules/lora.py`)
*   **Non-Invasive Architecture**: Uses a wrapper-based approach to inject `LoRALinear` layers into existing Attention modules (Q/K/V/Out) without modifying the core UNet source code.
*   **Dynamic Injection**: Features an `inject_lora` utility that recursively scans the model for `CrossAttention` and `Attention` blocks.
*   **Selective Freezing**: Includes `freeze_non_lora` logic to ensure >95% of the base model parameters are locked, significantly reducing VRAM consumption.

### 2. Offline Automated Captioning (`scripts/generate_captions.py`)
*   **BLIP-Large Integration**: Leverages the `Salesforce/blip-image-captioning-large` model for high-quality, automated image labeling.
*   **Bulk Processing**: Scans dataset directories and generates standardized `metadata.jsonl` files, automating the most tedious part of the fine-tuning process.

### 3. Enhanced Data Pipeline (`sgm/data/dataset.py`)
*   **Metadata Mapping**: Modified the `StableDataModuleFromConfig` to support an external `metadata_path`.
*   **Runtime Caption Injection**: Implemented a `map` operator within the data pipeline to dynamically pair offline-generated captions with training images during the setup phase.

### 4. Visual A/B Testing Tool (`scripts/demo/ab_test.py`)
*   **Streamlit UI**: A dedicated web-based dashboard for Side-by-Side (SbS) comparison.
*   **Synced Sampling**: Forces identical random seeds and prompts across Base and LoRA models to ensure a fair and scientific evaluation of fine-tuning quality.
*   **Memory Management**: Integrated `torch.cuda.empty_cache()` to allow sequential model swapping on consumer-grade GPUs (16GB-24GB).

---

## 🛠️ Retained Stability AI Core Features

The framework maintains 100% compatibility with the original Stability AI SGM features:

*   **Full SGM Architecture**: Native support for **SDXL Base 1.0**, **Refiner**, and **SVD (Stable Video Diffusion)**.
*   **Advanced GPU Optimization**:
    *   **Mixed Precision**: Full support for `fp16` and `bf16`.
    *   **Gradient Accumulation**: Compatible with `main.py` parameters for training on hardware with limited VRAM.
*   **Standardized Training Lifecycle**:
    *   **Monitoring**: Deep integration with **WandB (Weights & Biases)** and TensorBoard.
    *   **Resuming**: Native support for `--resume` to recover training from `.ckpt` files.
*   **Model Quality Assessment**:
    *   **FID Support**: Retains the `InceptionV3` wrapper in `sgm/modules/encoders/modules.py` for Fréchet Inception Distance calculations.
    *   **CLIP Foundation**: Native CLIP/OpenCLIP embedders are preserved for consistency.

---

## 📖 Local Workflow Guide

Since you are running this in a local environment (non-Docker), follow these steps:

### 1. Generate Dataset Metadata
Place your training images in `dataset/pokemon-images` and run:
```bash
python3 scripts/generate_captions.py
```

### 2. Launch LoRA Fine-tuning
Ensure your `configs/training/sdxl_pokemon_finetune.yaml` is correctly pointing to your local paths, then run:
```bash
python3 main.py --base configs/training/sdxl_pokemon_finetune.yaml --train --gpus 0,
```

### 3. Visual Evaluation (A/B Test)
Run the Streamlit dashboard to compare your results:
```bash
streamlit run scripts/demo/ab_test.py
```

---

## 📦 Required Local Dependencies
Ensure your local environment has the following packages installed:
*   `pip install streamlit transformers pytorch-lightning omegaconf safetensors`
*   `pip install bitsandbytes peft tqdm pillow xformers`

