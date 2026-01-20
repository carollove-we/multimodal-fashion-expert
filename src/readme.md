# 源代码文档 (Source Code Documentation)

本目录包含了多模态时尚助手（Multimodal Fashion Assistant）的核心实现代码，包括模型架构定义、训练流程、离线索引脚本以及 Streamlit Web 应用程序。

## 文件概览

| 文件名                           | 描述                                                                                                   |
| :------------------------------- | :----------------------------------------------------------------------------------------------------- |
| **`app_visual.py`**      | **[主程序]** Streamlit Web 界面，实现了“两阶段逻辑链推理”的核心逻辑。                          |
| **`model_arch_fast.py`** | **[模型定义]** 定义了 `FastLLaVA` 类 (LLM + Projector + LoRA) 和 `ResamplerProjector` 结构。 |
| **`train_projector.py`** | **[训练脚本]** 用于训练 Projector 连接层并通过 LoRA 微调 LLM 的主脚本。                          |
| **`precompute.py`**      | **[预处理]** 离线提取图像的 CLIP 特征，可提高训练速度                                            |
| **`pre_data_full.py`**   | **[预处理]** 清洗原始的 `styles.csv` 并生成训练专用的 `data_fast.json` 数据集。              |
| **`offline_indexer.py`** | **[搜索]** 构建用于图像检索的向量索引文件 (`image_embeddings.pt`)。                            |
| **`loss_plotter.py`**    | **[工具]** 用于绘制训练 Loss 曲线的辅助工具。                                                    |
| **`model_arch.py`**        | **[模型定义]** 早期版本的模型定义文件，已被 `model_arch_fast.py` 替代(但是`model_arch_fast.py`需要其中的交叉注意力模块)                          |
| **`CLIP_train.py`**      | **[训练脚本]** 用于微调 CLIP 模型以适应时尚图像的脚本。                                          |
| **`split_data.py`**      | **[预处理]** 将数据集划分为训练集和验证集的脚本。                                                |


---

## 训练指南

* **运行pre_data_full.py**
* **输入**: `../fashion-dataset/styles.csv` 和 `../fashion-dataset/images/`
* **输出**: `../fashion-dataset/data.json`
* **运行split_data.py分割数据集**
* **运行CLIP_train.py对模型进行微调**
* **运行precompute.py生成CLIP特征（注意每次调整CLIP后都要重新运行）**
* **运行train_projector.py对投影层训练**
* **运行offline_indexer.py生成离线索引，提高后续在streamlit页面上显示的速度**
* **在控制台输入 `streamlit run src/app_visual.py`(在streamlit之前可以配置config.tomel实现个性化页面）**
