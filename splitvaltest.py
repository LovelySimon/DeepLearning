import os
import shutil
import random


def split_dataset(src_dir, dest_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 确保比例和为 1
    assert train_ratio + val_ratio + test_ratio == 1, "The sum of the ratios must be 1."

    # 获取所有图片文件
    all_images = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    # 打乱文件顺序
    random.shuffle(all_images)

    # 计算每个集合的大小
    total_images = len(all_images)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - train_size - val_size

    # 创建目标文件夹
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dest_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

    # 将图片划分到对应的文件夹
    for i, image in enumerate(all_images):
        src_path = os.path.join(src_dir, image)
        if i < train_size:
            dest_path = os.path.join(dest_dir, 'train', image)
        elif i < train_size + val_size:
            dest_path = os.path.join(dest_dir, 'val', image)
        else:
            dest_path = os.path.join(dest_dir, 'test', image)

        shutil.copy(src_path, dest_path)

    print(f"Dataset split complete: {train_size} train, {val_size} val, {test_size} test.")


# 使用示例
src_directory = 'C://Users//Administrator//Desktop//data//yesi'  # 替换为你的图片文件夹路径
dest_directory = 'C://Users//Administrator//Desktop//data//yesi_1'  # 替换为保存划分后的数据集的目标文件夹路径
split_dataset(src_directory, dest_directory,0.8,0.1,0.1)