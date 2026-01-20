import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import json
from torch.optim import AdamW
from tqdm import tqdm
import random


from model_arch_fast import FastLLaVA
from loss_plotter import plot_loss_curve, plot_loss_with_moving_average, save_loss_history

# ================= 配置 =================
LLM_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_JSON = "fashion-dataset/data_fast.json" 
OUTPUT_DIR = "save/my_resampler_lora_v1" 

BATCH_SIZE = 8
ACCUMULATION_STEPS = 8
EPOCHS = 12     
LR = 1e-4      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 数据集 (加入 Visual Dropout 优化) =================
class MaskedInstructionDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        
        self.prompts = [
            "Describe this image.",
            "Analyze the visual style.",
            "What fashion item is this?",
            "Detail the material and color.",
        ]
        self.prefixes = ["It is ", "This shows ", "Here is ", ""]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        feature_path = item.get('feature_path')
        
        # 异常处理：找不到文件就换下一个
        if not feature_path or not os.path.exists(feature_path):
            return self.__getitem__((idx + 1) % len(self.data))
        try:
            image_features = torch.load(feature_path, map_location="cpu").squeeze(0)
        except:
            return self.__getitem__((idx + 1) % len(self.data))
        
        #  Visual Dropout: 20% 概率把图片抹零，强迫模型看文本
        if random.random() < 0.2:
            image_features = torch.zeros_like(image_features)
        
        # 构造 Prompt
        prompt = random.choice(self.prompts)
        caption = item['caption']
        prefix = random.choice(self.prefixes)
        target_text = f"{prefix}{caption}"
        
        # 编码
        bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else 1
        prompt_text = f"[INST] {prompt} [/INST] "
        prompt_ids = [bos] + self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target_text + self.tokenizer.eos_token, add_special_tokens=False).input_ids
        
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        
        # Padding
        max_len = 128
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
        else:
            pad_len = max_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            
        return {
            "image_features": image_features,
            "input_ids": torch.tensor(input_ids).long(),
            "attention_mask": torch.tensor([1 if i != self.tokenizer.pad_token_id else 0 for i in input_ids]).long(),
            "labels": torch.tensor(labels).long()
        }

# ================= 训练循环 =================
def train():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 初始化带 LoRA 的模型
    model = FastLLaVA(LLM_ID, device=device, use_lora=True)
    model.train()
    
    dataset = MaskedInstructionDataset(DATA_JSON, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    

    trainable_params = [
        {'params': model.projector.parameters(), 'lr': LR},
        {'params': filter(lambda p: p.requires_grad, model.llm.parameters()), 'lr': LR}
    ]
    optimizer = AdamW(trainable_params, weight_decay=0.01)
    
    total_steps = len(dataloader) * EPOCHS // ACCUMULATION_STEPS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.05), total_steps)
    scaler = torch.amp.GradScaler('cuda')
    
    print("Start LoRA Training...")
    

    loss_history = {'epochs': [], 'losses': []}
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            with torch.amp.autocast('cuda'):
                outputs = model(
                    batch['image_features'].to(device),
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['labels'].to(device)
                )
                loss = outputs.loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({"loss": f"{loss.item() * ACCUMULATION_STEPS:.4f}"})
            
        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_epoch_loss:.4f}")
        
        # 记录loss历史
        loss_history['epochs'].append(epoch + 1)
        loss_history['losses'].append(avg_epoch_loss)
        
 
        plot_loss_curve(loss_history, OUTPUT_DIR, save_name="loss_curve.png")
        plot_loss_with_moving_average(loss_history, OUTPUT_DIR, window_size=1, save_name="loss_curve_smooth.png")
        save_loss_history(loss_history, OUTPUT_DIR, save_name="loss_history.json")
        
        # 保存模型
        # if (epoch + 1) % 2 == 0: # 每2个epoch保存一次
        #     # 1. 保存 Projector
        #     torch.save(model.projector.state_dict(), os.path.join(OUTPUT_DIR, f"projector_epoch_{epoch+1}.pt"))
            
        #     # 2. 保存 LoRA
        #     # PeftModel 提供了 save_pretrained 方法
        #     lora_save_path = os.path.join(OUTPUT_DIR, f"lora_epoch_{epoch+1}")
        #     model.llm.save_pretrained(lora_save_path)
        #     print(f"Saved Projector & LoRA to {OUTPUT_DIR}")
    
    print("\nTraining completed! Loss curves saved.")

if __name__ == "__main__":
    train()