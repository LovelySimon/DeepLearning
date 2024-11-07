import os
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder
import random

transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
])

transform2 = transforms.Compose([
    transforms.RandomRotation(30),
])

transform3 = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])

transform4 = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
])
data_dir = '/home/wh603/桌面/carsclassification/train'  # 原始图像数据集路径
save_dir = '/home/wh603/桌面/carsclassification/train'  # 增强图像保存的根目录，通常可以和原始路径相同
dataset = ImageFolder(root=data_dir)
for i, (img, label) in enumerate(dataset):
    original_path = dataset.imgs[i][0]

    original_dir = os.path.dirname(original_path)

    # 获取原始图像文件名和扩展名
    img_name, img_ext = os.path.splitext(os.path.basename(original_path))

    # 使用增强操作
    augmented_image1 = transform1(img)
    augmented_image2 = transform2(img)
    augmented_image3 = transform3(img)
    augmented_image4 = transform4(img)

    # 创建增强后的图像文件名
    augmented_img_name1 = f"{img_name}_augmented_1{img_ext}"
    augmented_img_name2 = f"{img_name}_augmented_2{img_ext}"
    augmented_img_name3 = f"{img_name}_augmented_3{img_ext}"
    augmented_img_name4 = f"{img_name}_augmented_4{img_ext}"

    # 拼接增强图像的完整路径
    augmented_img_path1 = os.path.join(original_dir, augmented_img_name1)
    augmented_img_path2 = os.path.join(original_dir, augmented_img_name2)
    augmented_img_path3 = os.path.join(original_dir, augmented_img_name3)
    augmented_img_path4 = os.path.join(original_dir, augmented_img_name4)

    # 保存增强后的图像到原图文件夹
    augmented_image1.save(augmented_img_path1)
    augmented_image2.save(augmented_img_path2)
    augmented_image3.save(augmented_img_path3)
    augmented_image4.save(augmented_img_path4)

