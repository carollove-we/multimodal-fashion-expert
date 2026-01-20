import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from model_arch import ResamplerProjector, VISUAL_DIM 

class FastLLaVA(nn.Module):
    def __init__(self, llm_id, device="cuda", use_lora=True):
        super().__init__() 
        self.device = device
        
        # 1. 只有 LLM 
        print("Loading LLM (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            quantization_config=bnb_config,
            force_download=True,
        )
        self.llm.requires_grad_(False)
        
        # 2. 注入 LoRA
        if use_lora:
            
            peft_config = LoraConfig(
                r=16,           
                lora_alpha=32,  
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()

        # 3. Projector (Trainable)
        print("Initializing Projector...")
        self.projector = ResamplerProjector(visual_dim=VISUAL_DIM).to(device)
        
        self._init_projector_weights()

    def _init_projector_weights(self):
        nn.init.normal_(self.projector.latents, std=0.02)
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, image_features, input_ids, attention_mask, labels=None):
        # 1. Resampler 投影
        img_embeds = self.projector(image_features.to(self.device)).to(self.llm.dtype)
        # 2. 文本嵌入
        token_embeds = self.llm.get_input_embeddings()(input_ids)

        # 3. 拼接
        inputs_embeds = torch.cat([img_embeds, token_embeds], dim=1)

        # 4. 调整 Mask
        B, K, _ = img_embeds.shape
        visual_mask = torch.ones((B, K), device=self.device)
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # 5. 调整 Labels
        if labels is not None:
            visual_labels = torch.full((B, K), -100, device=self.device)
            labels = torch.cat([visual_labels, labels], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )