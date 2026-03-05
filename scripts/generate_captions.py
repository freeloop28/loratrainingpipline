import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# 配置
IMAGE_DIR = "dataset/pokemon-images"
OUTPUT_JSONL = os.path.join(IMAGE_DIR, "metadata.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_captions():
    print(f"Loading BLIP model on {DEVICE}...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images in {IMAGE_DIR}")
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for img_name in tqdm(image_files, desc="Generating Captions"):
            img_path = os.path.join(IMAGE_DIR, img_name)
            try:
                raw_image = Image.open(img_path).convert('RGB')
                inputs = processor(raw_image, return_tensors="pt").to(DEVICE)
                out = model.generate(**inputs)
                caption = processor.decode(out[0], skip_special_tokens=True)
                
                item = {"file_name": img_name, "text": caption}
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    print(f"Done! Metadata saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    generate_captions()
