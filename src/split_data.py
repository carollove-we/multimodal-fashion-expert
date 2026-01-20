import json
import os
import random

# 读取data.json
data_path = os.path.join(os.path.dirname(__file__), '../fashion-dataset/data.json')
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 设置随机种子以保证可重复性
random.seed(42)

# 随机打乱数据
shuffled_data = data.copy()
random.shuffle(shuffled_data)

# 分割数据为训练集(80%)和测试集(20%)
split_idx = int(len(shuffled_data) * 0.8)
train_data = shuffled_data[:split_idx]
test_data = shuffled_data[split_idx:]

# 保存训练集
train_output_path = os.path.join(os.path.dirname(__file__), '../fashion-dataset/train_data.json')
with open(train_output_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

# 保存测试集
test_output_path = os.path.join(os.path.dirname(__file__), '../fashion-dataset/test_data.json')
with open(test_output_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print(f"数据分割完成!")
print(f"总数据量: {len(data)}")
print(f"训练集: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
print(f"测试集: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
print(f"训练集已保存到: {train_output_path}")
print(f"测试集已保存到: {test_output_path}")
