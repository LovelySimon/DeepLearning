# 多分类感知机的实现
# 假设共有K个类别,则需要K-1条分界线,将某一类别j与其他类别进行划分
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
# 加载样本数据
# 为了实现多分类的可视化,在原来的数据基础上增加了一些数据,表示等待录取结果的学生,类别标记为0
data_x = pd.read_csv("./train/x.txt", sep=' ', header=None, names=['Exam 1', 'Exam 2'])
data_y = pd.read_csv("./train/y.txt", header=None, names=['Admitted'])
x = np.array(data_x.values)
y = np.array(data_y.values)
y1 = np.array(data_y.values)
y2 = np.array(data_y.values)
#数组y1中类别只有1和-1两种 将0与-1归为一类后再与类别1进行分类
y1[y1 < 1] = -1

#数组y2中将类别1和-1都标记为1，将类别0标记为-1
y2[y2 == -1] = 1
y2[y2 == 0] = -1

# 设定第一条分界线的初始值
w1 = [10, 17.5]  # 权值
b1 = -1400     # 偏置
n1 = 0.01  # 学习率
# 设定第二条分界线的初始值
w2 = [-6.8, 10]
b2 = -200
n2 = 0.01
# 第一次分类 将类别1与类别-1,0进行分类
def train1(x, y):
    # 当训练集中存在误分类点时
    global w1, b1
    #设置迭代次数为9999次 由于能够保证每次更新w,b值时误分类点个数比上一次少，因此次数尽量较大更贴合结果
    max = 9999
    for itr in range(max):
        # 随机选择一个样本
        rand = random.randint(0, 102)
        x0, x1, y1 = x[rand][0], x[rand][1], y[rand]
        # 当该随机点是误分类点时,判断更新后的w,b的值是否能使误分类点个数减少
        if (y1 * (w1[0] * x0 + w1[1] * x1 + b1) <= 0):
            ww = [0, 0]
            ww[0] = w1[0] + n1 * y1 * x0
            ww[1] = w1[1] + n1 * y1 * x1
            bb = b1 + n1 * y1
            #当误分类点个数减少时，更新w b的值
            if(count(x, y, ww, bb)<count(x, y, w1, b1)):
                w1[0] = ww[0]
                w1[1] = ww[1]
                b1 = bb
# 第二次分类 将类别0与类别+1,-1进行分类
def train2(x, y):
    # 当训练集中存在误分类点时
    global w2, b2
    max = 9999
    for itr in range(max):
        # 随机选择一个样本
        rand = random.randint(0, 102)
        x0, x1, y1 = x[rand][0], x[rand][1], y[rand]
        # 当该随机点是误分类点时,判断更新后的w,b的值是否能使误分类点个数减少
        if (y1 * (w2[0] * x0 + w2[1] * x1 + b2) <= 0):
            ww = [0, 0]
            ww[0] = w2[0] + n2 * y1 * x0
            ww[1] = w2[1] + n2 * y1 * x1
            bb = b2 + n2 * y1
            #当误分类点个数减少时，更新w b的值
            if(count(x, y, ww, bb)<count(x, y, w2, b2)):
                w2[0] = ww[0]
                w2[1] = ww[1]
                b2 = bb
# 计算误分类点个数的函数count
def count(x, y, ww, bb):
    ans = 0
    for i in range(0, len(x)):
        x0, x1, y1 = x[i][0], x[i][1], y[i]
        #对误分类点进行计数
        if (y1 * (ww[0] * x0 + ww[1] * x1 + bb) <= 0):
            ans += 1
    return ans

if __name__ == "__main__":
    # 调用训练函数,train1第一次分类 train2第二次分类
    train1(x, y1)
    train2(x, y2)
    # 输出最终训练结果
    print('first:')
    print('第一次分类权值wo为：', w1[0], '权值w1为：', w1[1], '偏置b为：', b1)
    print('second:')
    print('第二次分类权值wo为：', w2[0], '权值w1为：', w2[1], '偏置b为：', b2)
    #绘制分类后的图像
    # 设置绘图相关的默认参数
    plt.rcParams['figure.figsize'] = (12.0, 9.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.figure()
    positive = np.argwhere(y == 1)[:, 0]
    negative = np.argwhere(y == -1)[:, 0]
    unknown = np.argwhere(y == 0)[:, 0]
    # 绘制散点 用绿色和红色分别表示录取和未录取
    plt.plot(x[positive, 0], x[positive, 1], 'ko', color='green', linewidth=2, markersize=7, label='Admitted')
    plt.plot(x[negative, 0], x[negative, 1], 'ko', color='red', markersize=7, label='notAdmitted')
    plt.plot(x[unknown, 0], x[unknown, 1], 'ko', color='black', markersize=7, label='wait-confirm')
    # 直线函数 x1为x轴范围
    x1 = np.array([[10], [70]])
    x2 = (-b1 - w1[0]*x1)/w1[1]
    x11 = np.array([[10], [70]])
    x22 = (-b2 - w2[0] * x11) / w2[1]
    # 绘制多分类的分界线
    plt.plot(x1, x2)
    plt.plot(x11, x22)
    plt.legend(['Admitted', 'Not admitted', 'wait-confirm', 'Boundary-1', 'Boundary-2'], loc='lower right')
    plt.title('Classfication of mul-perceptron')
    plt.xlabel('Exam1')
    plt.xlabel('Exam2')
    plt.show()








