import os
import shutil
from pathlib import Path
import sys

# ================= 配置 =================
DATASET_NAME = "paramaggarwal/fashion-product-images-dataset"
TARGET_DIR = "fashion-dataset"

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("错误: 未找到 Kaggle 库。")
    print("请运行: pip install kaggle")
    sys.exit(1)

def setup_kaggle_api():
    """初始化 Kaggle API"""
    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except OSError:
        print(" 错误: 未找到 kaggle.json 配置文件。")
        print("请确保文件位于 ~/.kaggle/ (Linux/Mac) 或 C:\\Users\\用户名\\.kaggle\\ (Windows)")
        sys.exit(1)

def organize_structure(target_dir):
    """
    核心逻辑：
    1. 在下载的文件中递归寻找 'styles.csv'。
    2. 认定 'styles.csv' 所在的文件夹就是真正的数据源。
    3. 把该文件夹下的 'images' 文件夹和 'styles.csv' 移动到 target_dir 的最外层。
    """
    root_path = Path(target_dir)
    print(f" 正在扫描 {target_dir} 以整理结构...")

    # 1. 寻找 styles.csv 
    found_csvs = list(root_path.rglob("styles.csv"))
    
    if not found_csvs:
        print(" 警告: 下载似乎不完整，未找到 styles.csv！")
        return


    source_csv_path = found_csvs[0]
    source_folder = source_csv_path.parent  # styles.csv 所在的文件夹
    
    # 目标位置
    target_csv_path = root_path / "styles.csv"
    target_images_dir = root_path / "images"

    # 如果已经在正确位置，就不动
    if source_folder.resolve() == root_path.resolve():
        print(" 文件结构已经是正确的，无需移动。")
        return

    print(f" 发现数据深藏在: {source_folder}")
    print(" 正在将数据移动到根目录...")

    # 2. 移动 styles.csv
    shutil.move(str(source_csv_path), str(target_csv_path))
    print(f"   - 已移动 styles.csv")

    # 3. 移动 images 文件夹
    source_images_dir = source_folder / "images"
    if source_images_dir.exists():
        if target_images_dir.exists():
            print("   - 目标 images 文件夹已存在，正在清理旧数据...")
            shutil.rmtree(str(target_images_dir))
        
        shutil.move(str(source_images_dir), str(target_images_dir))
        print(f"   - 已移动 images 文件夹 (包含图片)")
    else:
        print(" 警告: 在 styles.csv 同级目录下未找到 images 文件夹！")

    # 4. 清理原本的深层空文件夹
    try:
        shutil.rmtree(str(source_folder)) # 删除那个之前的深层文件夹
        if source_folder.parent != root_path:
             shutil.rmtree(str(source_folder.parent), ignore_errors=True)
    except Exception as e:
        pass 

def main():
    print("=============================================")
    print("🛍️  Fashion Dataset Downloader (Auto-Fix)")
    print("=============================================")
    print(f"目标数据集: {DATASET_NAME}")
    print(f"本地保存目录: ./{TARGET_DIR}/")
    print("---------------------------------------------")

    # 1. 登录
    api = setup_kaggle_api()
    print(" Kaggle API 连接成功")

    # 2. 确保目录存在
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    # 3. 下载 (如果已存在 zip 不会重复下载，但会重新解压)
    print(" 开始下载/解压 (文件较大，请耐心等待)...")
    try:
        api.dataset_download_files(DATASET_NAME, path=TARGET_DIR, unzip=True)
        print("下载解压完成")
    except Exception as e:
        print(f"下载中断: {e}")
        sys.exit(1)

    # 4. 整理结构 
    organize_structure(TARGET_DIR)

    print("=============================================")
    print(" 准备就绪！")
    print("现在你的目录结构应该是：")
    print(f"  {TARGET_DIR}/")
    print(f"  ├── styles.csv")
    print(f"  └── images/")
    print(f"      ├── 1000.jpg")
    print(f"      └── ...")
    print("如果不是，请前往KAGGLE手动下载，并保留在fashion-dataset目录下。")
    print("数据集官网位于https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset")
    print("=============================================")

if __name__ == "__main__":
    main()