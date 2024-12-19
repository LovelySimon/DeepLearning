import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

from pip._internal.models import candidate

# 设置路径
image_folder = r'C:\Users\Administrator\Desktop\assignment\test_image'
coords_folder = r'C:\Users\Administrator\Desktop\assignment\test_txt\test_txt_50'
output_file = r'C:\Users\Administrator\Desktop\assignment\result.txt'

# 读取单个文本文件中的备选坐标数据
def load_coordinates_from_file(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            coords = tuple(map(int, parts))
            coordinates.append(coords)
    return coordinates

def preprocess_image(image):
    """进行图像预处理，避免重复计算"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return gray, edges

#水平度评分
def horizontal_score(image, candidate, gray):
    x1, y1, x2, y2 = candidate
    cropped = image[y1:y2, x1:x2]

    # 使用 Sobel 算子计算水平方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # 计算水平梯度的方向
    angle = np.arctan2(grad_x, np.ones_like(grad_x)) * 180 / np.pi

    # 计算与水平线的夹角
    # 目标是最接近水平方向，即夹角接近0度
    horizontal_angle_score = np.abs(angle.mean())  # 计算整个区域的平均角度

    # 评分：夹角越小，得分越高
    score = 1 / (horizontal_angle_score + 1)  # 防止除零错误，最小值设为1
    return score


#三分法
def rule_of_thirds_score(image, candidate, gray):
    x1, y1, x2, y2 = candidate
    w, h = x2 - x1, y2 - y1
    center_x, center_y = x1 + w // 2, y1 + h // 2

    thirds_x = [image.shape[1] // 3, 2 * image.shape[1] // 3]
    thirds_y = [image.shape[0] // 3, 2 * image.shape[0] // 3]

    distances = [abs(center_x - tx) + abs(center_y - ty) for tx in thirds_x for ty in thirds_y]
    score = 1.0 / (min(distances) + 1)  # 距离越小得分越高
    return score

#黄金比例
def golden_ratio_score(candidate):
    x1, y1, x2, y2 = candidate
    w, h = x2 - x1, y2 - y1
    golden_ratio = 1.618
    aspect_ratio = w / h if h != 0 else 0
    score = 1.0 / (abs(aspect_ratio - golden_ratio) + 1)  # 比例越接近黄金比例得分越高
    return score


def edge_density_score(image, candidate, edges):
    x1, y1, x2, y2 = candidate
    cropped_edges = edges[y1:y2, x1:x2]

    # 确保裁剪后的边缘图是二值图（在一些情况下，边缘图可能是灰度图）
    cropped_edges = np.where(cropped_edges > 0, 1, 0)  # 将所有非零值转换为1，其他为0

    # 计算边缘密度：边缘像素占总像素的比例
    edge_pixels = np.sum(cropped_edges)  # 边缘像素数量
    total_pixels = cropped_edges.size  # 总像素数量

    # 边缘密度 = 边缘像素 / 总像素
    edge_density = edge_pixels / total_pixels

    return edge_density


def vertical_score(image, candidate, gray):
    x1, y1, x2, y2 = candidate
    cropped = image[y1:y2, x1:x2]

    # 计算水平方向和垂直方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # 水平梯度
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # 垂直梯度

    # 计算每个像素的梯度方向
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi  # 计算梯度方向（弧度转为角度）

    # 提取裁剪区域的角度信息
    cropped_angle = angle[y1:y2, x1:x2]

    # 计算该区域内所有像素的平均角度偏离垂直方向的角度
    # 目标是最接近垂直方向，即 90 或 -90 度
    # 计算每个像素的与垂直方向的差异
    vertical_angle_diff = np.abs(cropped_angle - 90)  # 垂直方向是90度，计算与垂直的角度差

    # 计算平均角度差
    avg_vertical_angle_diff = np.mean(vertical_angle_diff)

    # 评分：角度差越小，垂直度越高，得分越高
    # 防止除零错误，最小值设为1
    score = 1 / (avg_vertical_angle_diff + 1)
    return score


def symmetry_score(image, candidate):
    x1, y1, x2, y2 = candidate
    cropped = image[y1:y2, x1:x2]  # 裁剪出候选区域

    # 计算该区域的水平对称轴
    cropped_width = cropped.shape[1]
    left_half = cropped[:, :cropped_width // 2]  # 左半部分
    right_half = cropped[:, cropped_width // 2:]  # 右半部分

    # 对右半部分进行水平翻转
    right_half_flipped = cv2.flip(right_half, 1)

    # 计算左右部分的均方误差（MSE）
    # 先调整尺寸使得左右部分对齐
    if left_half.shape != right_half_flipped.shape:
        right_half_flipped = cv2.resize(right_half_flipped, (left_half.shape[1], left_half.shape[0]))

    # 计算均方误差（MSE）
    mse = np.sum((left_half - right_half_flipped) ** 2) / float(left_half.size)

    # 评分：MSE 越小，表示越对称，评分越高
    symmetry_score = 1 / (mse + 1)  # 防止除零错误，最小值设为1
    return symmetry_score

# #评分函数（未归一化版本）
# def evaluate_candidate(image, candidate, gray, edges):
#     horizontal_score_val = horizontal_score(image, candidate, gray)
#     golden_ratio_score_val = golden_ratio_score(candidate)
#     edge_density_score_val = edge_density_score(image, candidate, edges)
#     vertical_score_val = vertical_score(image, candidate, gray)
#     symmetry_score_val = symmetry_score(image, candidate)
#
#     # 将评分组合成列表
#     scores = [horizontal_score_val, golden_ratio_score_val, edge_density_score_val, vertical_score_val ,
#               symmetry_score_val ]
#     return scores

# 得分计算函数(归一化版本)
def evaluate_candidate(image, candidate, gray, edges):
    # 获取各个评分
    horizontal_score_val = horizontal_score(image, candidate, gray)
    golden_ratio_score_val = golden_ratio_score(candidate)
    edge_density_score_val = edge_density_score(image, candidate, edges)
    vertical_score_val = vertical_score(image, candidate, gray)
    symmetry_score_val = symmetry_score(image, candidate)

    # 将评分组合成列表
    scores = [horizontal_score_val, golden_ratio_score_val, edge_density_score_val, vertical_score_val*100,
              symmetry_score_val*100]
    # 对每个评分进行归一化（例如，使用最小-最大归一化）
    normalized_scores = min_max_normalization(scores)
    return normalized_scores

def min_max_normalization(scores):
    # 使用 Min-Max 归一化方法，将评分缩放到 [0, 1] 范围
    min_score = min(scores)
    max_score = max(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]

def calculate_final_score(normalized_scores, weights=[0.3, 0.4, 0.1, 0.1, 0.1]):
    # 确保权重和为1
    total_score = sum(w * score for w, score in zip(weights, normalized_scores))
    return total_score

def process_image(image_path, coords_file_path, image, gray, edges):
    image_id = os.path.splitext(os.path.basename(coords_file_path))[0]
    candidates = load_coordinates_from_file(coords_file_path)

    # 如果没有候选框，则返回
    if not candidates:
        return None

    # 找到评分最高的候选框
    best_candidate = max(candidates, key=lambda c: calculate_final_score(evaluate_candidate(image, c, gray, edges)))
    return image_id, best_candidate


def process_images(image_folder, coords_folder, output_file):
    # 创建线程池来加速图像处理
    with ThreadPoolExecutor(max_workers=4) as executor, open(output_file, 'w') as f:
        futures = []

        for file_name in os.listdir(coords_folder):
            if file_name.endswith('.txt'):
                coords_file_path = os.path.join(coords_folder, file_name)
                image_id = os.path.splitext(file_name)[0]
                image_path = os.path.join(image_folder, f'{image_id}.jpg')

                # 加载图像
                image = cv2.imread(image_path)
                if image is None:
                    continue

                # 预处理图像
                gray, edges = preprocess_image(image)

                # 提交线程池进行候选框评估
                futures.append(executor.submit(process_image, image_path, coords_file_path, image, gray, edges))

        # 等待所有任务完成，并写入输出文件
        for future in futures:
            result = future.result()
            if result:
                image_id, best_candidate = result
                x1, y1, x2, y2 = best_candidate
                f.write(f'{image_id} {x1} {y1} {x2} {y2}\n')
                print(f'已写入结果: {image_id} {x1} {y1} {x2} {y2}')


# 执行图像处理并输出结果
process_images(image_folder, coords_folder, output_file)