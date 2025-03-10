import os
import shutil
import random

# 定义源文件夹和目标文件夹路径
source_folder = 'C://Users//alj//Desktop//seg_train'  # 原始数据所在文件夹
target_folder = 'C://Users//alj//Desktop//seg_new'  # 目标数据集文件夹

# 目标文件夹结构
jpeg_images_folder = os.path.join(target_folder, 'JPEGImages')
segmentation_class_folder = os.path.join(target_folder, 'SegmentationClass')
image_sets_folder = os.path.join(target_folder, 'ImageSets')
classes_file = os.path.join(target_folder, 'classes.txt')

# 创建目标文件夹
os.makedirs(jpeg_images_folder, exist_ok=True)
os.makedirs(segmentation_class_folder, exist_ok=True)
os.makedirs(image_sets_folder, exist_ok=True)
os.makedirs(os.path.join(image_sets_folder, 'Segmentation'), exist_ok=True)

# 获取类别文件夹列表
categories = ['single_1', 'single_2', 'single_3', 'single_4']

# 遍历每个类别文件夹
image_list = []
for category in categories:
    category_folder = os.path.join(source_folder, category)

    # 遍历类别文件夹中的每个文件
    for file_name in os.listdir(category_folder):
        if file_name.endswith('_1.png'):  # 只处理以 _1.png 结尾的原图文件
            source_path = os.path.join(category_folder, file_name)

            # 构建对应的掩码文件名
            mask_file_name = file_name.replace('_1.png', '_2.png')
            mask_source_path = os.path.join(category_folder, mask_file_name)

            # 检查掩码文件是否存在
            if os.path.exists(mask_source_path):
                # 构建新的文件名（在文件名前面加上类别前缀，避免重复）
                new_image_name = f"{category}_{file_name}"
                new_mask_name = f"{category}_{mask_file_name}"

                # 移动原图到 JPEGImages 文件夹
                image_target_path = os.path.join(jpeg_images_folder, new_image_name)
                shutil.copy(source_path, image_target_path)

                # 移动掩码到 SegmentationClass 文件夹
                mask_target_path = os.path.join(segmentation_class_folder, new_mask_name)
                shutil.copy(mask_source_path, mask_target_path)

                # 记录图像名称（不带扩展名）
                image_list.append(new_image_name.split('.')[0])
            else:
                print(f"Warning: Mask file {mask_file_name} not found for image {file_name}. Skipping...")

# 打乱数据顺序
random.shuffle(image_list)

# 生成 classes.txt 文件
with open(classes_file, 'w') as f:
    for category in categories:
        f.write(f"{category}\n")

# 生成 ImageSets/train.txt 和 ImageSets/val.txt 文件
train_val_split = 0.8  # 80% 用于训练，20% 用于验证
num_images = len(image_list)
train_images = image_list[:int(num_images * train_val_split)]
val_images = image_list[int(num_images * train_val_split):]

with open(os.path.join(image_sets_folder, 'Segmentation', 'train.txt'), 'w') as f:
    for image in train_images:
        f.write(f"{image}\n")

with open(os.path.join(image_sets_folder, 'Segmentation', 'val.txt'), 'w') as f:
    for image in val_images:
        f.write(f"{image}\n")