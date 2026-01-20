import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
import os
import torch
from transformers import CLIPModel
from tqdm import tqdm
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
import numpy as np

output_dir =os.path.join("save","fine_tuned_clip_model16")
num_epochs=40
batch_size=64  
model_name = "openai/clip-vit-base-patch16"
dataset_path = "fashion-dataset"
accumulation_steps = 4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class EcommerceDataset(Dataset):
    def __init__(self, json_file, processor,is_train=True):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        self.is_train = is_train
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert("RGB")
        text = item['caption']

        
        text_inputs = self.processor(text=[text], padding="max_length", truncation=True, return_tensors="pt")
        
        
        if self.is_train:
           
            pixel_values = self.train_transform(image)
        else:
            
            pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': pixel_values
        }
def train_one_epoch(model, dataloader, optimizer,scheduler, accumulation_steps=8):
    model.train()
    total_loss = 0
    optimizer.zero_grad() 

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        # 前向传播
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values,
            return_loss=True 
        )
        
        loss = outputs.loss
        loss_normalized = loss / accumulation_steps        
        loss_normalized.backward()
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()      
            scheduler.step()
            optimizer.zero_grad() 
        total_loss += loss.item()         
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")
    return avg_loss
def train(model,dataloader,epochs,optimizer,scheduler,processor):
    metrics_history = []
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1} / {epochs} ===")
        train_one_epoch(model, dataloader, optimizer, scheduler, accumulation_steps=accumulation_steps)
        r1, r5, r10 = evaluate(model, evaluate_dataloader)
        
        # 保存指标
        metrics_history.append({
            'epoch': epoch + 1,
            'r1': r1,
            'r5': r5,
            'r10': r10
        })
    
    # 绘制并保存图表
    plot_metrics(metrics_history)
    #model.save_pretrained(output_dir,safe_serialization=True)
    #processor.save_pretrained(output_dir)
def evaluate(model,dataloader): 
    model.eval() 
    all_image_embeds = []
    all_text_embeds = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 获取特征
            image_features = model.get_image_features(pixel_values)
            text_features = model.get_text_features(input_ids, attention_mask=attention_mask)

            
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            all_image_embeds.append(image_features)
            all_text_embeds.append(text_features)

    # 拼接所有 Batch
    all_image_embeds = torch.cat(all_image_embeds)
    all_text_embeds = torch.cat(all_text_embeds)

    print(f"计算相似度矩阵: {len(all_image_embeds)} x {len(all_text_embeds)}")
    logits_per_image = torch.matmul(all_image_embeds, all_text_embeds.t())


    n_samples = logits_per_image.shape[0]
    labels = torch.arange(n_samples).to(device)

    _, top1_indices = logits_per_image.topk(1, dim=1)
    _, top5_indices = logits_per_image.topk(5, dim=1)
    _, top10_indices = logits_per_image.topk(10, dim=1)


    r1 = (top1_indices == labels.view(-1, 1)).sum().item() / n_samples
    
    r5 = sum([label in indices for label, indices in zip(labels, top5_indices)]) / n_samples
    r10 = sum([label in indices for label, indices in zip(labels, top10_indices)]) / n_samples

    print("\n" + "="*30)
    print(f"评估结果 (Samples: {n_samples})")
    print(f"Recall@1:  {r1:.4f} ({(r1*100):.2f}%)")
    print(f"Recall@5:  {r5:.4f} ({(r5*100):.2f}%)")
    print(f"Recall@10: {r10:.4f} ({(r10*100):.2f}%)")
    print("="*30 + "\n")
    
    return r1, r5, r10


def plot_metrics(metrics_history):
    """绘制 R1、R5、R10 的变化曲线"""
    epochs = list(range(1, len(metrics_history) + 1))
    r1_values = [m['r1'] for m in metrics_history]
    r5_values = [m['r5'] for m in metrics_history]
    r10_values = [m['r10'] for m in metrics_history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, r1_values, marker='o', label='Recall@1', linewidth=2)
    plt.plot(epochs, r5_values, marker='s', label='Recall@5', linewidth=2)
    plt.plot(epochs, r10_values, marker='^', label='Recall@10', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recall Score', fontsize=12)
    plt.title('Model Recall Metrics Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join("save", "recall_metrics.png")
    os.makedirs("save", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\n图表已保存到: {save_path}")
    plt.show()


model=CLIPModel.from_pretrained(model_name,use_safetensors=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6,weight_decay=1e-2)
train_json_file = os.path.join(dataset_path, "train_data.json") 
train_dataset=EcommerceDataset(train_json_file, CLIPProcessor.from_pretrained(model_name, return_tensors="pt"))
train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
evaluate_json_file = os.path.join(dataset_path, "test_data.json") 
evaluate_dataset=EcommerceDataset(evaluate_json_file, CLIPProcessor.from_pretrained(model_name, return_tensors="pt"),is_train=False)
evaluate_dataloader=DataLoader(evaluate_dataset, batch_size=32, shuffle=False, num_workers=4)
steps_per_epoch = len(train_dataloader) // accumulation_steps
total_steps = steps_per_epoch * num_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1), 
    num_training_steps=total_steps
)
processor=CLIPProcessor.from_pretrained(model_name, return_tensors="pt")
train(model,train_dataloader,epochs=num_epochs,optimizer=optimizer,scheduler=scheduler,processor=processor)