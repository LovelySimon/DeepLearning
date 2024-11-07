import os
import shutil
import random

# 定义源文件夹和目标文件夹
source_dir = '/home/wh603/桌面/carsclassification/xiaopeng_p7'   # 源文件夹路径，包含所有图片
train_dir = '/home/wh603/桌面/carsclassification/train/class_5'    # 训练集保存的文件夹
test_dir = '/home/wh603/桌面/carsclassification/test/class_5'      # 测试集保存的文件夹

# 创建训练集和测试集的文件夹（如果它们不存在的话）
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 遍历源文件夹中的图片并划分
files = os.listdir(source_dir)

# 打乱文件顺序，确保数据分布随机
random.shuffle(files)

# 设置划分比例，例如 90% 为训练集，10% 为测试集
train_size = int(0.9 * len(files))
test_size = len(files) - train_size

# 将图片分为训练集和测试集
train_files = files[:train_size]
test_files = files[train_size:]

# 将图片移动到相应的文件夹
for file in train_files:
    source_path = os.path.join(source_dir, file)
    destination_path = os.path.join(train_dir, file)
    shutil.copy(source_path, destination_path)

for file in test_files:
    source_path = os.path.join(source_dir, file)
    destination_path = os.path.join(test_dir, file)
    shutil.copy(source_path, destination_path)

print(f"Training set size: {len(train_files)}")
print(f"Testing set size: {len(test_files)}")