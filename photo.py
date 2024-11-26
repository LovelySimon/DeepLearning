import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor


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

# 曲线检测评分（Curved算法）
def curved_score(image, candidate, gray , edges):
    x1, y1, x2, y2 = candidate
    cropped = image[y1:y2, x1:x2]

    # 先进行 Canny 边缘检测

    # 然后进行霍夫变换，尝试检测圆形或其他曲线
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=100)

    # 评估曲线的存在与数量
    if circles is not None:
        circles = np.uint16(np.around(circles))
        score = len(circles[0])  # 曲线数量越多，得分越高
    else:
        score = 0  # 没有检测到曲线，得分为0

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


#视觉权重
def visual_weight_score(image, candidate, gray):
    x1, y1, x2, y2 = candidate
    cropped = image[y1:y2, x1:x2]

    # 使用 Harris 角点检测来计算特征点数量
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    if corners is not None:
        score = len(corners)
    else:
        score = 0
    return score

#边缘密度
def edge_density_score(image, candidate,edges):
    x1, y1, x2, y2 = candidate
    cropped = image[y1:y2, x1:x2]

    # 边缘检测
    edge_density = np.sum(edges) / cropped.size
    return edge_density

#ROT区域
def rot_score(image, candidate, gray):
    x1, y1, x2, y2 = candidate

    # 人脸检测
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    overlap_score = 0
    for (fx, fy, fw, fh) in faces:
        # 判断候选框和检测到的人脸框的重叠度
        overlap_area = max(0, min(x2, fx + fw) - max(x1, fx)) * max(0, min(y2, fy + fh) - max(y1, fy))
        face_area = fw * fh
        if overlap_area > 0:
            overlap_score += overlap_area / face_area

    return overlap_score


#对角线算法评分
def diagonal_score(image,candidate):
    x1, y1, x2, y2 = candidate
    # 计算候选框的对角线长度
    diagonal_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 计算图像对角线
    image_diag_length = np.sqrt((image.shape[1]) ** 2 + (image.shape[0]) ** 2)

    # 计算对角线相对位置的评分
    score = 1 / (abs(diagonal_length - image_diag_length) + 1)
    return score


# 手动实现LBP
def lbp(image, radius=1, n_points=8):
    # 获取图像的尺寸
    height, width = image.shape

    # LBP结果
    lbp_image = np.zeros_like(image)

    # 遍历图像中的每个像素，排除边缘像素
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center_value = image[y, x]
            binary_values = []
            # 对每个邻域像素进行比较，形成二进制数
            for i in range(n_points):
                # 计算相邻像素的坐标
                angle = 2 * np.pi * i / n_points
                dx = int(radius * np.cos(angle))
                dy = int(radius * np.sin(angle))
                neighbor_value = image[y + dy, x + dx]
                binary_values.append(int(neighbor_value >= center_value))

            # 将二进制数转换为十进制数
            lbp_value = sum([binary_values[i] * 2 ** i for i in range(n_points)])
            lbp_image[y, x] = lbp_value

    return lbp_image

#纹理识别
def pattern_score(image, candidate , gray):
    x1, y1, x2, y2 = candidate
    cropped = image[y1:y2, x1:x2]

    # 转为灰度图像

    # 使用手动实现的LBP
    lbp1 = lbp(gray, radius=1, n_points=8)

    # 计算LBP直方图并归一化
    lbp_hist = cv2.calcHist([lbp1.astype(np.uint8)], [0], None, [256], [0, 256])
    lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()

    # 计算与图像中其他区域的相似度（例如使用相似度度量，如欧几里得距离）
    score = np.sum(lbp_hist)  # 直接使用纹理特征的总和作为评分
    return score

# 垂直度评分（Vertical算法）
def vertical_score(image, candidate , gray):
    x1, y1, x2, y2 = candidate
    cropped = image[y1:y2, x1:x2]


    # 使用 Sobel 算子计算水平和垂直方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # 计算梯度的方向
    angle = np.arctan2(grad_x, np.ones_like(grad_x)) * 180 / np.pi

    # 计算与垂直方向的夹角
    # 目标是最接近垂直，即夹角接近 0 度，越接近垂直，得分越高
    vertical_angle_score = np.abs(angle.mean())  # 计算整个区域的平均角度

    # 评分：夹角越小，得分越高
    score = 1 / (vertical_angle_score + 1)  # 防止除零错误，最小值设为1
    return score

def normalize_z_score(scores):
    """Z-score标准化，将评分转换为标准正态分布"""
    mean = np.mean(scores)
    std = np.std(scores)
    if std == 0:
        return scores  # 防止除以零
    return [(score - mean) / std for score in scores]

def normalize_min_max(scores, min_val=0.0, max_val=1.0):
    """最大最小归一化，将评分转换到 [min_val, max_val] 范围内"""
    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score
    if range_score == 0:
        return [min_val] * len(scores)  # 防止除以零
    return [(score - min_score) / range_score * (max_val - min_val) + min_val for score in scores]

# 得分计算函数
def evaluate_candidate(image, candidate, gray, edges):
    horizontal_score_val = horizontal_score(image, candidate, gray)
    rot_score_val = rule_of_thirds_score(image, candidate, gray)
    golden_ratio_score_val = golden_ratio_score(candidate)
    visual_weight_score_val = visual_weight_score(image, candidate, gray)
    edge_density_score_val = edge_density_score(image, candidate,edges)
    pattern_score_val = pattern_score(image, candidate , gray)
    diagonal_score_val = diagonal_score(image,candidate)
    vertical_score_val = vertical_score(image, candidate , gray)

    scores = [horizontal_score_val, rot_score_val, golden_ratio_score_val,visual_weight_score_val,edge_density_score_val,pattern_score_val,diagonal_score_val,vertical_score_val]
    normalized_scores = normalize_z_score(scores)
    # 将各个得分归一化
    total_score = sum(normalized_scores)

    # 计算每个算法的得分占比
    horizontal_score_val_weight=normalized_scores[0]/total_score
    rot_score_val_weight=normalized_scores[1]/total_score
    golden_ratio_score_val_weight=normalized_scores[2]/total_score
    visual_weight_score_val_weight=normalized_scores[3]/total_score
    edge_density_score_val_weight=normalized_scores[4]/total_score
    pattern_score_val_weight=normalized_scores[5]/total_score
    diagonal_score_val_weight=normalized_scores[6]/total_score
    vertical_score_val_weight=normalized_scores[7]/total_score

    # 根据得分动态调整权重
    # 给得分较高的算法更多的权重
    max_score=max(horizontal_score_val_weight,rot_score_val_weight,golden_ratio_score_val_weight,visual_weight_score_val_weight,edge_density_score_val_weight,pattern_score_val_weight,diagonal_score_val_weight,vertical_score_val_weight)

    # 动态加权，得分高的算法权重加倍
    weights = {
        'horizontal': horizontal_score_val_weight * (1 + horizontal_score_val_weight / max_score),
        'rot': rot_score_val_weight * (1 + rot_score_val_weight / max_score),
        'golden_ratio': golden_ratio_score_val_weight * (1 + golden_ratio_score_val_weight / max_score),
        'visual_weight': visual_weight_score_val_weight * (1 + visual_weight_score_val_weight / max_score),
        'edge_density': edge_density_score_val_weight * (1 + edge_density_score_val_weight / max_score),
        'pattern': pattern_score_val_weight * (1 + pattern_score_val_weight / max_score),
        'diagonal': diagonal_score_val_weight * (1 + diagonal_score_val_weight / max_score),
        'vertical': vertical_score_val_weight * (1 + vertical_score_val_weight / max_score)
    }

    # 计算总得分，所有评分的加权和
    weighted_score = (
            weights['horizontal'] * normalized_scores[0] +
            weights['rot'] * normalized_scores[1] +
            weights['golden_ratio'] * normalized_scores[2] +
            weights['visual_weight'] * normalized_scores[3] +
            weights['edge_density'] * normalized_scores[4] +
            weights['pattern'] * normalized_scores[5] +
            weights['diagonal'] * normalized_scores[6] +
            weights['vertical'] * normalized_scores[7]
    )

    return weighted_score

def process_image(image_path, coords_file_path, image, gray, edges):
    image_id = os.path.splitext(os.path.basename(coords_file_path))[0]
    candidates = load_coordinates_from_file(coords_file_path)

    # 如果没有候选框，则返回
    if not candidates:
        return None

    # 找到评分最高的候选框
    best_candidate = max(candidates, key=lambda c: evaluate_candidate(image, c, gray, edges))
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