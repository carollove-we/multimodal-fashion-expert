# Multimodal Fashion Assistant (Logic-Chain Enhanced)

[![Powered by Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-Mistral--7B%20%2B%20CLIP-blue)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

A state-of-the-art multimodal fashion search system that solves the "Visual Dominance" problem using a **Two-Stage Logic-Chain Inference** strategy. It allows users to modify visual queries with text (e.g., "Make it red") and generates high-end fashion editor descriptions.

## 📂 Directory Structure

```text
.
├── fashion-dataset/           # Dataset images and styles.csv
├── offline_data/              # Generated search indices (image_embeddings.pt)
├── save/                      # Model checkpoints
│   ├── fine_tuned_clip_model16/
│   └── my_resampler_lora_v1/  # Your trained Projector & LoRA
├── src/                       # Source Code
│   ├── app_visual.py          # Main Streamlit App
│   ├── model_arch_fast.py     # Model Definitions
│   ├── train_projector.py     # Training Script
│   ├── offline_indexer.py     # Indexing Script
│   └── ...
└── requirements.txt           # Dependencies
```

通过download.py下载数据集，推荐通过kaggle官网下载，数据集名称为fashion product Images Dataset
