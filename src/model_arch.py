import torch
import torch.nn as nn
from transformers import CLIPModel, AutoModelForCausalLM, BitsAndBytesConfig


NUM_VISUAL_TOKENS = 32 # K = 32


VISUAL_DIM = 768        
LLM_DIM = 4096          # Mistral-7B 的隐藏层维度

class ResamplerProjector(nn.Module):
    """
    使用 Cross-Attention 将视觉特征重采样为固定数量 (K) 的 Token。
    """
    def __init__(self, visual_dim=VISUAL_DIM, llm_dim=LLM_DIM, num_queries=NUM_VISUAL_TOKENS, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        

        self.latents = nn.Parameter(torch.randn(num_queries, llm_dim))
        

        self.vis_proj = nn.Linear(visual_dim, llm_dim)
        
        # 3. Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 4. FFN & Norm
        self.layer_norm = nn.LayerNorm(llm_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim)
        )

    def forward(self, visual_features):
        """
        Args:
            visual_features: [Batch, Seq_Len(Patches), Vis_Dim]
        Returns:
            [Batch, K, LLM_Dim]
        """
        B = visual_features.shape[0]
        
        # 映射视觉特征
        x = self.vis_proj(visual_features) # [B, Patches, LLM_Dim]
        
        # 扩展 Latents
        latents = self.latents.repeat(B, 1, 1) # [B, K, LLM_Dim]
        
        # Cross Attention: Query=Latents, Key/Value=Visual
        attn_out, _ = self.cross_attn(query=latents, key=x, value=x)
        
        # Residual + FFN
        latents = self.layer_norm(latents + attn_out)
        latents = latents + self.ffn(latents)
        
        return latents

class CustomLLaVA(nn.Module):
    def __init__(self, clip_path, llm_id, device="cuda"):
        super().__init__()
        self.device = device
        
        # A. 视觉塔 (Frozen)
        print("Loading CLIP...")
        self.clip = CLIPModel.from_pretrained(clip_path).to(device)
        self.clip.vision_model.requires_grad_(False)
        
        # B. 语言塔 (Frozen, 4-bit)
        print("Loading LLM...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            quantization_config=bnb_config,
        )
        self.llm.requires_grad_(False)
        
        # C. Projector (Trainable)
        print("Initializing Resampler Projector...")
        self.projector = ResamplerProjector().to(device)
        
        # 初始化权重
        nn.init.normal_(self.projector.latents, std=0.02)
        self.projector.vis_proj.apply(self._init_weights)
        self.projector.ffn.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):

        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values)
            image_features = vision_outputs.last_hidden_state 


       

        img_embeds = self.projector(image_features).to(self.llm.dtype)

        
        inputs_embeds = self.llm.model.embed_tokens(input_ids)

        #  拼接策略
        # [Image_Tokens, Text_Tokens]
        inputs_embeds = torch.cat([img_embeds, inputs_embeds], dim=1)

        #  调整 Mask
        B, K, _ = img_embeds.shape
        visual_mask = torch.ones((B, K), device=self.device)
        attention_mask = torch.cat([visual_mask, attention_mask], dim=1)

        if labels is not None:
            visual_labels = torch.full((B, K), -100, device=self.device)
            labels = torch.cat([visual_labels, labels], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )