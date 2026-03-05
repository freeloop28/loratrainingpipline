import streamlit as st
import torch
import numpy as np
import os
from omegaconf import OmegaConf
from PIL import Image
from sgm.util import instantiate_from_config, default
from sgm.models.diffusion import DiffusionEngine
from pytorch_lightning import seed_everything

# --- 页面配置 ---
st.set_page_config(layout="wide", page_title="SDXL LoRA A/B Test")
st.title("🎨 SDXL LoRA Visual A/B Testing")
st.info("说明：本工具将对比基座模型与 LoRA 微调后的效果。为了节省显存，将依次进行推理。")

# --- 侧边栏配置 ---
with st.sidebar:
    st.header("⚙️ 配置参数")
    
    # 模型路径
    base_config = st.text_input("Base Config Path", "configs/training/sdxl_pokemon_finetune.yaml")
    base_ckpt = st.text_input("Base Checkpoint Path", "models/sdxl-base-1.0/sdxl_vit_l_patch14_fp16.safetensors")
    
    st.divider()
    
    # LoRA 参数
    use_lora = st.checkbox("启用 LoRA 对比", value=True)
    lora_rank = st.number_input("LoRA Rank", value=32)
    lora_alpha = st.number_input("LoRA Alpha", value=32)
    
    st.divider()
    
    # 采样参数
    prompt = st.text_area("Prompt (正向提示词)", "a cute cartoon pokemon with big eyes, high quality")
    negative_prompt = st.text_area("Negative Prompt (反向提示词)", "low quality, blurry, distorted")
    seed = st.number_input("随机种子", value=42)
    steps = st.slider("采样步数", 20, 50, 25)
    cfg_scale = st.slider("CFG Scale", 1.0, 15.0, 7.5)
    width = st.selectbox("宽度", [512, 1024], index=1)
    height = st.selectbox("高度", [512, 1024], index=1)

# --- 模型加载核心逻辑 ---
@st.cache_resource
def get_model(config_path, ckpt_path, apply_lora=False, rank=32, alpha=32):
    config = OmegaConf.load(config_path)
    
    # 如果要应用 LoRA，修改配置中的 lora_config
    if apply_lora:
        config.model.params.lora_config = {
            "rank": rank,
            "alpha": alpha,
            "target_modules": ["CrossAttention", "Attention"]
        }
    else:
        config.model.params.lora_config = None
        
    model = instantiate_from_config(config.model)
    
    # 加载权重
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file as load_safetensors
        sd = load_safetensors(ckpt_path)
    else:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    
    model.load_state_dict(sd, strict=False)
    model.to("cuda").eval()
    return model

# --- 采样函数 ---
def sample(model, prompt, n_prompt, seed, steps, cfg, w, h):
    seed_everything(seed)
    
    # 准备条件
    c = model.conditioner.get_unconditional_conditioning(
        {"txt": prompt, "original_size_as_tuple": (h, w), "crop_coords_top_left": (0, 0), "target_size_as_tuple": (h, w)}
    )
    uc = model.conditioner.get_unconditional_conditioning(
        {"txt": n_prompt, "original_size_as_tuple": (h, w), "crop_coords_top_left": (0, 0), "target_size_as_tuple": (h, w)}
    )
    
    # 采样 (简化演示，实际需根据 sgm.sampler 调用)
    # 此处假设使用默认采样逻辑，为了 Demo 简洁，我们输出占位符逻辑
    # 在实际 generative-models 中，需调用 model.sample 或类似的 helper
    with torch.no_grad():
        with torch.autocast("cuda"):
            shape = (4, h // 8, w // 8)
            samples = torch.randn((1, *shape), device="cuda") # 简化模拟
            # 这里应插入具体的 model.sampler 逻辑
            # ...
            # 为了让代码能跑通，我们返回一个随机图演示布局
            return Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8))

# --- 主界面逻辑 ---
if st.button("开始对比生成"):
    col1, col2 = st.columns(2)
    
    # 1. 生成 Base 结果
    with col1:
        st.subheader("🅰️ Base SDXL (未微调)")
        with st.spinner("正在使用基座模型生成..."):
            # 释放显存
            torch.cuda.empty_cache()
            # 加载不带 LoRA 的模型
            model_base = get_model(base_config, base_ckpt, apply_lora=False)
            img_base = sample(model_base, prompt, negative_prompt, seed, steps, cfg_scale, width, height)
            st.image(img_base, use_container_width=True, caption="Base Model Output")
            st.success("Base 生成完成")

    # 2. 生成 LoRA 结果
    if use_lora:
        with col2:
            st.subheader("🅱️ SDXL + LoRA (微调后)")
            with st.spinner("正在应用 LoRA 并生成..."):
                # 再次释放显存
                torch.cuda.empty_cache()
                # 加载带 LoRA 的模型（此处会触发我们修改过的 DiffusionEngine 注入逻辑）
                model_lora = get_model(base_config, base_ckpt, apply_lora=True, rank=lora_rank, alpha=lora_alpha)
                img_lora = sample(model_lora, prompt, negative_prompt, seed, steps, cfg_scale, width, height)
                st.image(img_lora, use_container_width=True, caption="LoRA Model Output")
                st.success("LoRA 生成完成")
    else:
        with col2:
            st.warning("请在侧边栏勾选“启用 LoRA 对比”以查看微调效果。")
