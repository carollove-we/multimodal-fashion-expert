import pandas as pd
import os
import json
import math

DATASET_ROOT = "fashion-dataset" 
OUTPUT_JSON = "fashion-dataset/data.json"


def clean_text(text):

    if pd.isna(text) or text == "nan":
        return ""
    return str(text).strip()

def process_full_dataset():
    csv_path = os.path.join(DATASET_ROOT, "styles.csv")
    images_dir = os.path.join(DATASET_ROOT, "images")

    if not os.path.exists(csv_path):
        print(f" 错误: 找不到 {csv_path}")
        print("请确保 DATASET_ROOT 设置正确。")
        return

    print(f" 正在读取 CSV: {csv_path} ...")
    # on_bad_lines='skip': 跳过 CSV 中格式错误的行（完整版数据集中有不少坏行）
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        print(f" CSV 读取失败: {e}")
        return

    print(f"{len(df)}")
    print(" 开始构建图文对...")

    valid_data = []
    missing_images = 0
    
    # 遍历每一行
    for index, row in df.iterrows():
        try:
            # 1. 找图片
            item_id = str(row['id'])
            image_filename = item_id + ".jpg"
            image_path = os.path.join(images_dir, image_filename)
            
            # 严格检查图片是否存在
            if not os.path.exists(image_path):
                missing_images += 1
                continue
            
            # 2. 提取字段并清洗
            gender = clean_text(row['gender'])
            master_cat = clean_text(row['masterCategory'])
            sub_cat = clean_text(row['subCategory'])
            article_type = clean_text(row['articleType'])
            color = clean_text(row['baseColour'])
            season = clean_text(row['season'])
            usage = clean_text(row['usage'])
            name = clean_text(row['productDisplayName'])
            
            # 处理年份 (有些是 2012.0 这种 float，转成 2012)
            year = row['year']
            if pd.notna(year):
                try:
                    year_str = str(int(float(year)))
                except:
                    year_str = ""
            else:
                year_str = ""

        
            

            desc_parts = [color, gender, master_cat]
            
            if sub_cat and sub_cat != master_cat:
                desc_parts.append(sub_cat)
            

            if article_type:
                desc_parts.append(f"({article_type})")
            

            if usage:
                desc_parts.append(f"for {usage}")
                
            if season or year_str:
                time_info = f"{season} {year_str}".strip()
                if time_info:
                    desc_parts.append(f", {time_info}")

            attr_text = " ".join(filter(None, desc_parts))
            
            final_caption = f"A photo of {attr_text}. {name}"
            
            final_caption = " ".join(final_caption.split())


            valid_data.append({
                "image": image_path, # 存绝对路径
                "caption": final_caption
            })

            if len(valid_data) % 5000 == 0:
                print(f"已处理 {len(valid_data)} 条...")

        except Exception as e:
            # 某一行出错不影响整体
            continue


    print("-" * 30)
    print(f"原始记录: {len(df)}")
    print(f"图片缺失: {missing_images}")
    print(f"有效数据: {len(valid_data)}")
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=4, ensure_ascii=False)
        
    print(f"已保存至: {OUTPUT_JSON}")


if __name__ == "__main__":
    process_full_dataset()