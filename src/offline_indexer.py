import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import json
import os
from tqdm import tqdm

# ================= 配置区 =================
CLIP_MODEL_PATH = "save/fine_tuned_clip_model16" 
DATA_JSON = "fashion-dataset/data.json"
OUTPUT_INDEX_FILE = "offline_data/image_embeddings.pt"  
OUTPUT_DATA_FILE = "offline_data/indexed_data.json"      
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_offline_index():
    print(f" 开始构建离线索引...")
    print(f"Loading CLIP from {CLIP_MODEL_PATH}...")
    
    # 加载模型
    try:
        model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, use_safetensors=True).to(device)
    except:
        model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    model.eval()

    # 加载原始数据
    with open(DATA_JSON, 'r') as f:
        raw_dataset = json.load(f)
    
    print(f"原始数据量: {len(raw_dataset)}")
    
    valid_data = []    
    image_embeddings = [] 
    batch_size = 128
    
    # 开启进度条
    with torch.no_grad():
        for i in range(0, len(raw_dataset), batch_size):
            batch = raw_dataset[i:i+batch_size]
            current_batch_images = []
            current_batch_meta = []
            
            # 1. 尝试加载图片 (过滤坏图)
            for item in batch:
                try:
                    img = Image.open(item['image']).convert("RGB")
                    current_batch_images.append(img)
                    current_batch_meta.append(item)
                except Exception as e:

                    continue
            
            if not current_batch_images:
                continue

            # 2. 批量计算特征
            try:
                inputs = processor(images=current_batch_images, return_tensors="pt", padding=True)
                img_feats = model.get_image_features(inputs['pixel_values'].to(device))
                img_feats /= img_feats.norm(p=2, dim=-1, keepdim=True)
                
                image_embeddings.append(img_feats.cpu())
                valid_data.extend(current_batch_meta)
            except Exception as e:
                print(f"Batch processing error: {e}")
                continue
            
            print(f"已处理 {len(valid_data)} / {len(raw_dataset)}...", end="\r")

    # 3. 拼接并保存
    if image_embeddings:
        final_index = torch.cat(image_embeddings) # [N, 512]
        
        print(f"\n 索引构建完成")
        print(f"最终有效数据: {len(valid_data)}")
        print(f"保存向量至: {OUTPUT_INDEX_FILE}")
        torch.save(final_index, OUTPUT_INDEX_FILE)
        
        print(f"保存元数据至: {OUTPUT_DATA_FILE}")
        with open(OUTPUT_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(valid_data, f, ensure_ascii=False, indent=2)
            
    else:
        print(" 错误: 没有生成任何有效索引")

if __name__ == "__main__":
    build_offline_index()