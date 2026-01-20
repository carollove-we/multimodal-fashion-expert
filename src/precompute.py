import torch
import json
import os
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

# ================= 配置 =================
CLIP_PATH = "save/fine_tuned_clip_model16"
DATA_JSON = "fashion-dataset/data.json"
OUTPUT_DIR = "offline_features"          
NEW_JSON = "fashion-dataset/data_fast.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def precompute():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f" 加载 CLIP 模型: {CLIP_PATH}")
    try:
        clip_model = CLIPModel.from_pretrained(CLIP_PATH, use_safetensors=True).vision_model.to(device)
    except:
        clip_model = CLIPModel.from_pretrained(CLIP_PATH).vision_model.to(device)
        
    clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)
    clip_model.eval()

    print(f" 读取数据: {DATA_JSON}")
    with open(DATA_JSON, 'r') as f:
        data = json.load(f)

    new_data = []
    
    print(" 开始提取特征")
    with torch.no_grad():
        for item in tqdm(data):
            image_path = item['image']
        
            base_name = os.path.basename(image_path)
            feature_filename = f"{os.path.splitext(base_name)[0]}.pt"
            feature_path = os.path.join(OUTPUT_DIR, feature_filename)
            
            if not os.path.exists(feature_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = clip_processor(images=image, return_tensors="pt").to(device)
                    
                    outputs = clip_model(**inputs)
                    last_hidden_state = outputs.last_hidden_state.cpu()
                    

                    torch.save(last_hidden_state, feature_path)
                except Exception as e:
                    print(f"跳过 {image_path}: {e}")
                    continue
            

            new_item = item.copy()
            new_item['feature_path'] = feature_path
            new_data.append(new_item)

    # 保存新的 JSON
    with open(NEW_JSON, 'w') as f:
        json.dump(new_data, f, indent=2)
    

if __name__ == "__main__":
    precompute()