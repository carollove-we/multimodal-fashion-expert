# ================= 基础依赖 =================
import streamlit as st
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import PeftModel
from PIL import Image
import json
import traceback
import re

from model_arch import ResamplerProjector, VISUAL_DIM

# ================= 配置 =================
CLIP_MODEL_PATH = "save/fine_tuned_clip_model16"
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# 指向你训练好的 Projector 和 LoRA
PROJECTOR_WEIGHT = "save/my_resampler_lora_v1/projector_epoch_6.pt" 
LORA_WEIGHT_DIR = "save/my_resampler_lora_v1/lora_epoch_6"          

INDEX_FILE = "offline_data/image_embeddings.pt"
DATA_FILE = "offline_data/indexed_data.json"
DATASET_ROOT = "fashion-dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 提示词  =================

SYSTEM_PROMPT_QUERY = """
You are a smart fashion search assistant.
Rule:
1. Identify CATEGORY from image (Shoe/Apparel).
2. Use COLOR from user text.
3. Combine: [User Color] [Material/Style] [Category].
Output ONE phrase.
"""


SYSTEM_PROMPT_SALES = """
You are a Senior Fashion Editor and Stylist.
Your task is to write a captivating, high-end product description for the item shown in the image.

STRICT GUIDELINES:
1. **Visual First**: Look closely at the image. Describe the texture (e.g., matte leather, distressed denim), the cut (e.g., slim fit, chunky sole), and unique details (e.g., stitching, logos).
2. **No Robot Talk**: NEVER start with "A photo of", "This image shows", or "I see". Start directly with the product's appeal (e.g., "Step up your game with...", "This stunning jacket features...").
3. **Expand & Enrich**: Do not just repeat the provided tags. Use your fashion knowledge to explain WHY these features are good.
4. **Styling Advice**: Always include a tip on how to wear it (e.g., "Pair this with...").

Structure:
**The Look**: [Vivid visual description based on the image]
**Why We Love It**: [Selling points and benefits]
**Style Tip**: [Outfit recommendation]

Write a passionate and professional recommendation.
"""

# ================= 系统加载 =================
@st.cache_resource
def load_system():
    # 1. 加载 CLIP
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载基础 LLM (4-bit)
    print("Loading Base LLM...")
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto"
    ).eval()
    
    # 3. 加载 LoRA 权重
    if os.path.exists(LORA_WEIGHT_DIR):
        print(f" Loading LoRA adapters from {LORA_WEIGHT_DIR}...")
        llm = PeftModel.from_pretrained(llm, LORA_WEIGHT_DIR)
    else:
        print(" Warning: LoRA weights not found! Running with frozen base model.")

    # 4. 加载 Projector
    projector = ResamplerProjector(VISUAL_DIM).to(device)
    if os.path.exists(PROJECTOR_WEIGHT):
        print(f"Loading Projector from {PROJECTOR_WEIGHT}...")
        projector.load_state_dict(torch.load(PROJECTOR_WEIGHT, map_location=device))
    else:
        print(f" Warning: Projector weights not found at {PROJECTOR_WEIGHT}")
        
    projector.eval()

    # 5. 索引数据
    if os.path.exists(INDEX_FILE):
        image_index = torch.load(INDEX_FILE, map_location=device)
        with open(DATA_FILE) as f:
            dataset = json.load(f)
        print(f" Loaded index with {len(dataset)} items.")
    else:
        image_index = None
        dataset = []
        print(" Warning: Offline index not found. Please run offline_indexer.py")

    return clip_model, clip_processor, llm, tokenizer, projector, image_index, dataset

clip_model, clip_processor, llm, tokenizer, projector, image_index, dataset = load_system()

# ================= 工具函数 =================
def smart_find_image(path):
    if not path: return None
    fname = os.path.basename(path)
    for p in [
        path,
        os.path.join(DATASET_ROOT, "images", fname),
        os.path.join(DATASET_ROOT, fname),
    ]:
        if os.path.exists(p):
            return p
    return None

def clean_query(query):
    query = re.sub(r'^(Output|Query|Answer|Result)[:\s\-]*', '', query, flags=re.IGNORECASE)
    query = query.replace('"', '').replace("'", "").strip()
    return query

# ================= ARCH STEP 1: Query 生成 =================
def generate_query(image, text):
    prompt = f"[INST]{SYSTEM_PROMPT_QUERY}\nUser Request: {text}[/INST]"

    visual_embeds = None
    if image:
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        feats = clip_model.vision_model(inputs.pixel_values).last_hidden_state
        visual_embeds = projector(feats).to(llm.dtype)
        

    text_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    text_embeds = llm.get_input_embeddings()(text_inputs.input_ids)

    if visual_embeds is not None:
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        attn = torch.cat([
            torch.ones((1, visual_embeds.shape[1]), device=device),
            text_inputs.attention_mask
        ], dim=1)
    else:
        inputs_embeds = text_embeds
        attn = text_inputs.attention_mask

    out = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=30,
        do_sample=False
    )
    raw_query = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return clean_query(raw_query)

# ================= ARCH STEP 2: CLIP 检索 =================
def retrieve_topk_pairs(query_text=None, query_image=None, k=3):
    if image_index is None: return []
    
    with torch.no_grad():
        if query_text:
            inputs = clip_processor(text=[query_text], return_tensors="pt").to(device)
            q = clip_model.get_text_features(**inputs)
        else:
            inputs = clip_processor(images=query_image, return_tensors="pt").to(device)
            q = clip_model.get_image_features(**inputs)

        q = q / q.norm(dim=-1, keepdim=True)
        sims = (q @ image_index.T).squeeze(0)
        idxs = sims.topk(k).indices.tolist()
        return [dataset[i] for i in idxs]

# ================= ARCH STEP 3: 导购生成 (重点优化) =================
def generate_sales_from_pairs(items):
    results = []
    for item in items:
        path = smart_find_image(item["image"])
        if not path: continue
        image = Image.open(path).convert("RGB")

        # 清洗 Caption
        raw_caption = item.get("caption", "")
        clean_tags = raw_caption.replace("A photo of", "").replace(".", ",").strip()
        prompt = f"[INST]{SYSTEM_PROMPT_SALES}\nProduct Metadata Tags: {clean_tags}[/INST]"

        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        feats = clip_model.vision_model(inputs.pixel_values).last_hidden_state
        visual_embeds = projector(feats).to(llm.dtype)

        text_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        text_embeds = llm.get_input_embeddings()(text_inputs.input_ids)

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        attn = torch.cat([
            torch.ones((1, visual_embeds.shape[1]), device=device),
            text_inputs.attention_mask
        ], dim=1)

        out = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=300, 
            min_new_tokens=100,
            temperature=0.8,   
            do_sample=True,
            top_p=0.9
        )
        recommendation = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        
        
        if "[/INST]" in recommendation:
            recommendation = recommendation.split("[/INST]")[-1].strip()
            
        
        if recommendation.lower().startswith("a photo of"):
            recommendation = recommendation[10:].strip()
            
        results.append({
            "item": item,
            "text": recommendation
        })
    return results

# ================= Streamlit UI =================
st.set_page_config(layout="wide")
st.title(" Multimodal Fashion Assistant ")

if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.history:
    for idx, record in enumerate(st.session_state.history):
        with st.container():
            st.markdown("---")
            col_img, col_query = st.columns([1, 3])
            with col_img:
                if record["image"]:
                    st.image(record["image"], width=120, caption="Query Image")
            with col_query:
                st.markdown(f"<h3 style='color:#1f77b4;'>Generated Query</h3>", unsafe_allow_html=True)
                if "query_used" in record:
                    st.code(record["query_used"], language="text")
                if record["text"]:
                    st.caption(f"User Instruction: {record['text']}")
            
            st.divider()
            
            st.markdown(f"<h3 style='color:#ff7f0e;'>Expert Recommendations</h3>", unsafe_allow_html=True)
            for r in record["results"]:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(smart_find_image(r["item"]["image"]), width=150)
                with col2:
                    st.markdown(r["text"]) 
            st.write("")
        st.markdown("---")

st.divider()

col_plus, col_center, col_search = st.columns([0.5, 4, 0.8])

with col_plus:
    st.write("") 
    if st.button("➕", key="plus_btn", help="New search"):
        st.session_state.history = []
        st.rerun()

with col_center:
    st.write("")
    text = st.text_input("Describe what you want", label_visibility="collapsed", placeholder="Example: 'Make it red' or 'I want blue ones'")
    col_space, col_img = st.columns([3, 1])
    with col_img:
        img = st.file_uploader("Upload image", type=["jpg", "png"], label_visibility="collapsed")

with col_search:
    st.write("")
    search_btn = st.button("🔍", key="search_btn", help="Search")

if search_btn:
    if not img and not text:
        st.error("Please upload an image or enter a description")
    else:
        try:
            image = Image.open(img).convert("RGB") if img else None
            query_used = ""

            if image and text:
                with st.spinner("Generating Query..."):
                    query = generate_query(image, text)
                    query_used = query
                with st.spinner("Retrieving items..."):
                    pairs = retrieve_topk_pairs(query_text=query)
            
            else:
                with st.spinner("Retrieving items..."):
                    pairs = retrieve_topk_pairs(
                        query_text=text if text else None,
                        query_image=image if image else None
                    )
                    query_used = text if text else "(Image Similarity Search)"

            with st.spinner("Generating Expert Reviews..."):
                results = generate_sales_from_pairs(pairs)
            
            st.session_state.history.append({
                "image": img,
                "text": text,
                "query_used": query_used,
                "results": results
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(traceback.format_exc())