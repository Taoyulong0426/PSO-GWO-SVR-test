# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:35:27 2023

@author: pony
"""

# 导入所需的库
import pandas as pd  # 数据处理
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图
import seaborn as sns  # 数据可视化
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归模型
import sklearn  # 机器学习库
from sklearn.metrics import mean_squared_error, r2_score  # 评估指标
import os  # 操作系统相关
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score  # 数据划分
from sklearn.preprocessing import StandardScaler  # 特征归一化
import random  # 随机数生成
import warnings  # 警告处理
import math

from woa import WOA
from gwo import GWO
from Igwo import IGWO
from pso import PSO
from psogwo import PSOGWO

np.random.seed(34)  # 设置随机种子

plt.rcParams['figure.figsize'] = (6.5, 13)
plt.rcParams.update({'font.size': 17})
plt.rcParams['font.sans-serif'] = ['simSun']  # 添加中文字体为黑体
# 在局部加上fontfamily="Times New Roman"
# plt.xticks(fontfamily="Times New Roman")
plt.rcParams['axes.unicode_minus'] = False

# 普通circle混吨映射
position = np.zeros(1000)

# 迭代计算映射值
Iteration = []
for i in range(1000):
    if i == 0:
        position[i] = random.random()
    else:
        position[i] = (position[i - 1] + 0.2 - 0.5 / (2 * math.pi) * math.sin(
            2 * math.pi * position[i - 1])) % 1
    Iteration.append(i)

# 绘制图
# 直方图
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].hist(position, bins=10, edgecolor='white')
ax[0].set_title('(a)  Histogram', y=-0.25)
ax[0].set_xlabel("Chaotic value")
ax[0].set_ylabel("Iteration")
# 散点图
ax[1].scatter(Iteration, position, c='red', s=5)
ax[1].set_title('(b)  Scatter plot', y=-0.25)
ax[1].set_xlabel("Number of iterations")
ax[1].set_ylabel("Chaotic value")
# 保存图片SVG格式
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.savefig('Circle直方图和散点图.svg')
plt.show()

# 改进Ciecle混吨映射
position = np.zeros(1000)

# 迭代计算映射值
Iteration = []
for i in range(1000):
    if i == 0:
        position[i] = random.random()
    else:
        position[i] = (4 * position[i - 1] + 0.5 - 0.8 / (2 * math.pi) * math.sin(
            2 * math.pi * position[i - 1])) % 1
    Iteration.append(i)

# 绘制图
# 直方图
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].hist(position, bins=10, edgecolor='white')
ax[0].set_title('(a)  Histogram', y=-0.25)
ax[0].set_xlabel("Chaotic value")
ax[0].set_ylabel("Iteration")
# 散点图
ax[1].scatter(Iteration, position, c='red', s=5)
ax[1].set_title('(b)  Scatter plot', y=-0.25)
ax[1].set_xlabel("Number of iterations")
ax[1].set_ylabel("Chaotic value")
# 保存图片SVG格式
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.savefig('改进后的Circle直方图和散点图.svg')
plt.show()

# # 普通tent混吨映射
# position = np.zeros(1000)
# a =5
# # 迭代计算映射值
# Iteration = []
# for i in range(1000):
#     if i == 0:
#         position[i] = random.random()
#     else:
#         position[i] = math.sin(math.pi*((a-1)*position[i-1]+(a-2)*math.cos(math.pi*position[i-1])))
#     Iteration.append(i)
#
# # 绘制图
# # 直方图
# fig, ax = plt.subplots(nrows=2, ncols=1)
# ax[0].hist(position, bins=10, edgecolor='white')
# ax[0].set_title('(a)  直方图', y=-0.3)
# ax[0].set_xlabel("混沌值")
# ax[0].set_ylabel("次数")
# # 散点图
# ax[1].scatter(Iteration, position, c='red', s=5)
# ax[1].set_title('(b)  散点图', y=-0.3)
# ax[1].set_xlabel("迭代次数")
# ax[1].set_ylabel("混沌值")
# # 保存图片SVG格式
# plt.subplots_adjust(wspace=0.2, hspace=0.3)
# plt.savefig('Circle直方图和散点图.svg')
# plt.show()
#
# # 改进Ciecle混吨映射
# position = np.zeros(1000)
#
# # 迭代计算映射值
# Iteration = []
# a = 0.6
# for i in range(1000):
#     if i == 0:
#         position[i] = random.random()
#     else:
#         if position[i - 1] < a:
#             position[i] = position[i - 1] / a
#         else:
#             position[i] = (1 - position[i - 1]) / (1 - a)
#     Iteration.append(i)
#
# # 绘制图
# # 直方图
# fig, ax = plt.subplots(nrows=2, ncols=1)
# ax[0].hist(position, bins=10, edgecolor='white')
# ax[0].set_title('(a)  直方图', y=-0.3)
# ax[0].set_xlabel("混沌值")
# ax[0].set_ylabel("次数")
# # 散点图
# ax[1].scatter(Iteration, position, c='red', s=5)
# ax[1].set_title('(b)  散点图', y=-0.3)
# ax[1].set_xlabel("迭代次数")
# ax[1].set_ylabel("混沌值")
# # 保存图片SVG格式
# plt.subplots_adjust(wspace=0.2, hspace=0.3)
# plt.savefig('改进后的Circle直方图和散点图.svg')
# plt.show()

# 列名定义
RUL_name = ['RUL']  # 寿命的名字
eigenvalue_names = ['Vol_cut_time', 'Tem_avr', 'V_col_dec', 'vol_avr', 'cur_avr']  # 特征值名字
col_names = RUL_name + eigenvalue_names  # 第一列到最后一列的名字

# 读取数据集
dftrain = pd.read_csv('./NASA_database/B0005.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取训练集
dfvalid = pd.read_csv('./NASA_database/B0005.csv', sep=',', header=None, index_col=False, names=col_names)  # 读取测试集

# 复制数据集
train = dftrain.copy()
valid = dfvalid.head(50).copy()

# 输出训练集中的缺失值数量
print('训练集中的缺失值数量：', train.isna().sum())

# 对数据归一化处理
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.preprocessing import MinMaxScaler  # 导入特征缩放器

scaler = MinMaxScaler()  # 创建MinMaxScaler实例用于特征缩放

X_train, X_test, y_train, y_test = train_test_split(train, train['RUL'], test_size=0.3,
                                                    random_state=42)  # 划分训练和测试数据集

# 删除目标变量
X_train.drop(columns=['RUL'], inplace=True)
X_test.drop(columns=['RUL'], inplace=True)
# 缩放X_train和X_test特征
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.fit_transform(X_test)

y_valid = valid['RUL']
valid.drop(columns=['RUL'], inplace=True)
X_valid_s = scaler.fit_transform(valid)


class Linear_Regression():
    def __init__(self, lr=0.01, iterations=150):
        self.lr = lr
        self.iterations = iterations

    def fit(self, X, Y):
        self.l, self.p = X.shape
        # 权重初始化
        self.W = np.zeros(self.p)
        self.b = 0
        self.X = X
        self.Y = Y
        # 梯度学习
        for i in range(self.iterations):
            self.weight_updater()
        return self

    def weight_updater(self):
        Y_pred = self.predict(self.X)
        # 计算梯度
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.l
        db = - 2 * np.sum(self.Y - Y_pred) / self.l
        # 更新权重
        self.b = self.b - self.lr * db
        self.W = self.W - self.lr * dW
        return self

    def predict(self, X):
        # Y_pred = X.W + b
        return X.dot(self.W) + self.b


from sklearn.svm import SVR
import tensorflow as tf

regressor = SVR(kernel='rbf', C=1, gamma=8)  # 创建SVR回归器

rf = RandomForestRegressor(max_features="sqrt", random_state=42)  # 创建随机森林回归器

# '''
# 线性拟合LR模型
# '''
# lr = Linear_Regression()  # 创建线性回归模型实例
# lr.fit(X=X_train_s, Y=y_train)  # 拟合模型
# print('LR模型:')
# y_lr_train = lr.predict(X_train_s)  # 在训练数据上进行预测
# evaluate(y_train, y_lr_train, label='train')
#
# y_lr_test = lr.predict(X_test_s)  # 在测试数据上进行预测
# evaluate(y_test, y_lr_test, label='test')
#
# y_lr_valid = lr.predict(X_valid_s)  # 在验证数据上进行预测
# evaluate(y_valid, y_lr_valid, label='valid')
#
# # 创建一LR验证集的图形
# plt.figure(figsize=(10, 6))
#
# # 绘制真实值的折线
# plt.plot(y_valid, label='真实值', marker='o', linestyle='-')
#
# # 绘制预测值的折线
# plt.plot(y_lr_valid , label='预测值', marker='x', linestyle='-')
# plt.title('LR验证集真实值与预测值')
# plt.show()
# # 创建一个LR测试集的显示图形
# plt.figure(figsize=(20, 6))
# y_test=np.array(y_test)
# # 绘制真实值的折线
# plt.plot(y_test[0:200], label='真实值', marker='o', linestyle='-')
# # 绘制预测值的折线
# plt.plot(y_lr_test[0:200] , label='预测值', marker='x', linestyle='-')
# plt.title('LR测试集真实值与预测值')
# plt.show()


plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams.update({'font.size': 14})


def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def Ufun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y


def fitness1(x):
    s = np.sum(x ** 2)
    return s


# [-100,100]

def fitness2(x):
    o = sum(abs(x)) + prod(abs(x))
    return o


# [-100,100]

def fitness3(x):
    dim = len(x)
    o = -20 * np.exp(-.2 * np.sqrt(np.sum(x ** 2) / dim)) - \
        np.exp(np.sum(np.cos(2 * math.pi * x)) / dim) + 20 + np.exp(1)
    return o


# [-32,32]

def fitness4(x):
    dim = len(x)
    w = [i for i in range(dim)]
    w = [i + 1 for i in w]
    o = np.sum(x ** 2) / 4000 - prod(np.cos(x / np.sqrt(w))) + 1
    return o


# [-600,600]


def plot(iterations_gwo_1, iterations_pso_1, iterations_woa_1, iterations_psogwo_1, accuracy_gwo_1, accuracy_pso_1,
         accuracy_woa_1, accuracy_psogwo_1,
         iterations_gwo_2, iterations_pso_2, iterations_woa_2, iterations_psogwo_2, accuracy_gwo_2, accuracy_pso_2,
         accuracy_woa_2, accuracy_psogwo_2,
         iterations_gwo_3, iterations_pso_3, iterations_woa_3, iterations_psogwo_3, accuracy_gwo_3, accuracy_pso_3,
         accuracy_woa_3, accuracy_psogwo_3,
         iterations_gwo_4, iterations_pso_4, iterations_woa_4, iterations_psogwo_4, accuracy_gwo_4, accuracy_pso_4,
         accuracy_woa_4, accuracy_psogwo_4
         ):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ysticks = [-330, -300, -200, -100, 1]
    yticks_labels = ['0', '$10^{-300}$', '$10^{-200}$', '$10^{-100}$', '$10^0$']  # 替换成你想要的标签
    ax1.plot(iterations_gwo_1, accuracy_gwo_1, label='GWO')
    ax1.plot(iterations_pso_1, accuracy_pso_1, label='PSO')
    ax1.plot(iterations_woa_1, accuracy_woa_1, label='WOA')
    ax1.plot(iterations_psogwo_1, accuracy_psogwo_1, label='FPSOGWO')
    ax1.set_yticks(ysticks, yticks_labels)
    ax1.set_ylim(-330, 1)
    ax1.set_xlim(0, 500)
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('f$_{min}$', fontsize=20)
    ax1.set_title('（a） F1', y=-0.23)
    ax1.legend(['IGWO', 'PSO', 'WOA', 'FPSOGWO'])  # 设置折线名称

    ysticks = [-330, -300, -200, -100, 1]
    yticks_labels = ['0', '$10^{-300}$', '$10^{-200}$', '$10^{-100}$', '$10^0$']  # 替换成你想要的标签
    ax2.plot(iterations_gwo_2, accuracy_gwo_2, label='GWO')
    ax2.plot(iterations_pso_2, accuracy_pso_2, label='PSO')
    ax2.plot(iterations_woa_2, accuracy_woa_2, label='WOA')
    ax2.plot(iterations_psogwo_2, accuracy_psogwo_2, label='FPSOGWO')
    ax2.set_yticks(ysticks, yticks_labels)
    ax2.set_ylim(-330, 1)
    ax2.set_xlim(0, 500)
    ax2.set_xlabel('Number of Iterations')
    ax2.set_ylabel('f$_{min}$', fontsize=20)
    ax2.set_title('（b） F2', y=-0.23)
    ax2.legend(['IGWO', 'PSO', 'WOA', 'FPSOGWO'])  # 设置折线名称

    ysticks = [-18, -16, -10, -4, 1]
    yticks_labels = ['0', '$10^{-16}$', '$10^{-10}$', '$10^{-4}$', '$10^0$']  # 替换成你想要的标签
    ax3.plot(iterations_gwo_3, accuracy_gwo_3, label='GWO')
    ax3.plot(iterations_pso_3, accuracy_pso_3, label='PSO')
    ax3.plot(iterations_woa_3, accuracy_woa_3, label='WOA')
    ax3.plot(iterations_psogwo_3, accuracy_psogwo_3, label='FPSOGWO')
    ax3.set_yticks(ysticks, yticks_labels)
    ax3.set_ylim(-18, 1)
    ax3.set_xlim(0, 500)
    ax3.set_xlabel('Number of Iterations')
    ax3.set_ylabel('f$_{min}$', fontsize=20)
    ax3.set_title('（C） F3', y=-0.23)
    ax3.legend(['IGWO', 'PSO', 'WOA', 'FPSOGWO'])  # 设置折线名称

    ysticks = [-18, -16, -10, -4, 1]
    yticks_labels = ['0', '$10^{-16}$', '$10^{-10}$', '$10^{-4}$', '$10^0$']  # 替换成你想要的标签
    ax4.plot(iterations_gwo_4, accuracy_gwo_4, label='GWO')
    ax4.plot(iterations_pso_4, accuracy_pso_4, label='PSO')
    ax4.plot(iterations_woa_4, accuracy_woa_4, label='WOA')
    ax4.plot(iterations_psogwo_4, accuracy_psogwo_4, label='FPSOGWO')
    ax4.set_yticks(ysticks, yticks_labels)
    ax4.set_ylim(-18, 1)
    ax4.set_xlim(0, 500)
    ax4.set_xlabel('Number of Iterations')
    ax4.set_ylabel('f$_{min}$', fontsize=20)
    ax4.set_title('（d） F4', y=-0.23)
    ax4.legend(['IGWO', 'PSO', 'WOA', 'FPSOGWO'])  # 设置折线名称
    plt.subplots_adjust(wspace=0.38, hspace=0.25)

    plt.savefig('英文四个函数下不同模型最优值变化.svg')
    plt.show()


# 参数
SearchAgents_no = 30  # 狼群数量
T = 500  # 最大迭代次数
dim = 1  # 寻最优参数个数
lb = [-100]
ub = [100]

init_w = 0.1
k = 0.4

# GWO优化
# 函数1的迭代
gwo_1 = GWO(fitness1, SearchAgents_no, T, dim, lb, ub)
iterations_gwo_1, accuracy_gwo_1 = gwo_1.sanitized_gwo()
GWO_best_1 = accuracy_gwo_1[499]
for i in range(len(iterations_gwo_1)):
    if accuracy_gwo_1[i] > 0:
        accuracy_gwo_1[i] = np.log(accuracy_gwo_1[i]) / np.log(10)

# 函数2的迭代
lb = [-10]
ub = [10]
gwo_2 = GWO(fitness2, SearchAgents_no, T, dim, lb, ub)
iterations_gwo_2, accuracy_gwo_2 = gwo_2.sanitized_gwo()
GWO_best_2 = accuracy_gwo_2[499]
for i in range(len(iterations_gwo_2)):
    if accuracy_gwo_2[i] > 0:
        accuracy_gwo_2[i] = np.log(accuracy_gwo_2[i]) / np.log(10)

# 函数3的迭代
lb = [-32]
ub = [32]
gwo_3 = GWO(fitness3, SearchAgents_no, T, dim, lb, ub)
iterations_gwo_3, accuracy_gwo_3 = gwo_3.sanitized_gwo()
GWO_best_3 = accuracy_gwo_3[499]
for i in range(len(iterations_gwo_3)):
    if accuracy_gwo_3[i] > 0:
        accuracy_gwo_3[i] = np.log(accuracy_gwo_3[i]) / np.log(10)

# 函数4的迭代
lb = [-600]
ub = [600]
gwo_4 = GWO(fitness4, SearchAgents_no, T, dim, lb, ub)
iterations_gwo_4, accuracy_gwo_4 = gwo_4.sanitized_gwo()
GWO_best_4 = accuracy_gwo_4[499]
for i in range(len(iterations_gwo_4)):
    if accuracy_gwo_4[i] > 0:
        accuracy_gwo_4[i] = np.log(accuracy_gwo_4[i]) / np.log(10)
    else:
        accuracy_gwo_4[i] = -18

# IGWO优化
# 函数1的迭代
Igwo_1 = IGWO(fitness1, SearchAgents_no, T, dim, lb, ub)
iterations_Igwo_1, accuracy_Igwo_1 = Igwo_1.sanitized_gwo()
IGWO_best_1 = accuracy_Igwo_1[499]
for i in range(len(iterations_Igwo_1)):
    if accuracy_Igwo_1[i] > 0:
        accuracy_Igwo_1[i] = np.log(accuracy_Igwo_1[i]) / np.log(10)

# 函数2的迭代
lb = [-10]
ub = [10]
Igwo_2 = IGWO(fitness2, SearchAgents_no, T, dim, lb, ub)
iterations_Igwo_2, accuracy_Igwo_2 = Igwo_2.sanitized_gwo()
IGWO_best_2 = accuracy_Igwo_2[499]
for i in range(len(iterations_Igwo_2)):
    if accuracy_Igwo_2[i] > 0:
        accuracy_Igwo_2[i] = np.log(accuracy_Igwo_2[i]) / np.log(10)

# 函数3的迭代
lb = [-32]
ub = [32]
Igwo_3 = IGWO(fitness3, SearchAgents_no, T, dim, lb, ub)
iterations_Igwo_3, accuracy_Igwo_3 = Igwo_3.sanitized_gwo()
IGWO_best_3 = accuracy_Igwo_3[499]
for i in range(len(iterations_Igwo_3)):
    if accuracy_Igwo_3[i] > 0:
        accuracy_Igwo_3[i] = np.log(accuracy_Igwo_3[i]) / np.log(10)

# 函数4的迭代
lb = [-600]
ub = [600]
Igwo_4 = IGWO(fitness4, SearchAgents_no, T, dim, lb, ub)
iterations_Igwo_4, accuracy_Igwo_4 = Igwo_4.sanitized_gwo()
IGWO_best_4 = accuracy_Igwo_4[499]
for i in range(len(iterations_Igwo_4)):
    if accuracy_Igwo_4[i] > 0:
        accuracy_Igwo_4[i] = np.log(accuracy_Igwo_4[i]) / np.log(10)
    else:
        accuracy_Igwo_4[i] = -18

# PSO优化
lb = [-100]
ub = [100]
# 函数1的迭代
pso_1 = PSO(fitness1, SearchAgents_no, T, dim, lb, ub)
iterations_pso_1, accuracy_pso_1 = pso_1.main()
PSO_best_1 = accuracy_pso_1[499]
for i in range(len(accuracy_pso_1)):
    if accuracy_pso_1[i] > 0:
        accuracy_pso_1[i] = np.log(accuracy_pso_1[i]) / np.log(10)
    else:
        accuracy_pso_1[i] = -330

# 函数2的迭代
lb = [-10]
ub = [10]
pso_2 = PSO(fitness2, SearchAgents_no, T, dim, lb, ub)
iterations_pso_2, accuracy_pso_2 = pso_2.main()
PSO_best_2 = accuracy_pso_2[499]
for i in range(len(accuracy_pso_2)):
    if accuracy_pso_2[i] > 0:
        accuracy_pso_2[i] = np.log(accuracy_pso_2[i]) / np.log(10)
    else:
        accuracy_pso_2[i] = -330

# 函数3的迭代
lb = [-32]
ub = [32]
pso_3 = PSO(fitness3, SearchAgents_no, T, dim, lb, ub)
iterations_pso_3, accuracy_pso_3 = pso_3.main()
PSO_best_3 = accuracy_pso_3[499]
for i in range(len(accuracy_pso_3)):
    if accuracy_pso_3[i] > 0:
        accuracy_pso_3[i] = np.log(accuracy_pso_3[i]) / np.log(10)
    else:
        accuracy_pso_3[i] = -330

# 函数4的迭代
lb = [-600]
ub = [600]
pso_4 = PSO(fitness4, SearchAgents_no, T, dim, lb, ub)
iterations_pso_4, accuracy_pso_4 = pso_4.main()
PSO_best_4 = accuracy_pso_4[499]
for i in range(len(accuracy_pso_4)):
    if accuracy_pso_4[i] > 0:
        accuracy_pso_4[i] = np.log(accuracy_pso_4[i]) / np.log(10)
    else:
        accuracy_pso_4[i] = -330

# woa优化
lb = [-100]
ub = [100]
# 函数1的迭代
b = 1
woa_1 = WOA(fitness1, SearchAgents_no, T, dim, lb, ub, b)
iterations_woa_1, accuracy_woa_1 = woa_1.opt()
WOA_best_1 = accuracy_woa_1[499]
for i in range(len(accuracy_woa_1)):
    if accuracy_woa_1[i] > 0:
        accuracy_woa_1[i] = np.log(accuracy_woa_1[i]) / np.log(10)
    else:
        accuracy_woa_1[i] = -330

# 函数2的迭代
lb = [-10]
ub = [10]
woa_2 = WOA(fitness2, SearchAgents_no, T, dim, lb, ub, b)
iterations_woa_2, accuracy_woa_2 = woa_2.opt()
WOA_best_2 = accuracy_woa_2[499]
for i in range(len(accuracy_woa_2)):
    if accuracy_woa_2[i] > 0:
        accuracy_woa_2[i] = np.log(accuracy_woa_2[i]) / np.log(10)
    else:
        accuracy_woa_2[i] = -330

# 函数3的迭代
lb = [-32]
ub = [32]
woa_3 = WOA(fitness3, SearchAgents_no, T, dim, lb, ub, b)
iterations_woa_3, accuracy_woa_3 = woa_3.opt()
WOA_best_3 = accuracy_woa_3[499]
for i in range(len(accuracy_woa_3)):
    if accuracy_woa_3[i] > 0:
        accuracy_woa_3[i] = np.log(accuracy_woa_3[i]) / np.log(10)
    else:
        accuracy_woa_3[i] = -330

# 函数4的迭代
lb = [-600]
ub = [600]
woa_4 = WOA(fitness4, SearchAgents_no, T, dim, lb, ub, b)
iterations_woa_4, accuracy_woa_4 = woa_4.opt()
WOA_best_4 = accuracy_woa_4[499]
for i in range(len(accuracy_woa_4)):
    if accuracy_woa_4[i] > 0:
        accuracy_woa_4[i] = np.log(accuracy_woa_4[i]) / np.log(10)
    else:
        accuracy_woa_4[i] = -330

# psogwo
lb = [-100]
ub = [100]
# 函数1的迭代
psogwo_1 = PSOGWO(fitness1, dim, SearchAgents_no, T, ub, lb,
                  init_w, k)
iterations_psogwo_1, accuracy_psogwo_1 = psogwo_1.opt()
PSOGWO_best_1 = accuracy_psogwo_1[499]
for i in range(len(accuracy_psogwo_1)):
    if accuracy_psogwo_1[i] > 0:
        accuracy_psogwo_1[i] = np.log(accuracy_psogwo_1[i]) / np.log(10)
    else:
        accuracy_psogwo_1[i] = -330

# 函数2的迭代
lb = [-10]
ub = [10]
psogwo_2 = PSOGWO(fitness2, dim, SearchAgents_no, T, ub, lb,
                  init_w, k)
iterations_psogwo_2, accuracy_psogwo_2 = psogwo_2.opt()
PSOGWO_best_2 = accuracy_psogwo_2[499]
for i in range(len(accuracy_psogwo_2)):
    if accuracy_psogwo_2[i] > 0:
        accuracy_psogwo_2[i] = np.log(accuracy_psogwo_2[i]) / np.log(10)
    else:
        accuracy_psogwo_2[i] = -330

# 函数3的迭代
lb = [-32]
ub = [32]
psogwo_3 = PSOGWO(fitness3, dim, SearchAgents_no, T, ub, lb,
                  init_w, k)
iterations_psogwo_3, accuracy_psogwo_3 = psogwo_3.opt()
PSOGWO_best_3 = accuracy_psogwo_3[499]
for i in range(len(accuracy_psogwo_3)):
    if accuracy_psogwo_3[i] > 0:
        accuracy_psogwo_3[i] = np.log(accuracy_psogwo_3[i]) / np.log(10)

# 函数4的迭代
lb = [-600]
ub = [600]
psogwo_4 = PSOGWO(fitness4, dim, SearchAgents_no, T, ub, lb,
                  init_w, k)
iterations_psogwo_4, accuracy_psogwo_4 = psogwo_4.opt()
PSOGWO_best_4 = accuracy_psogwo_4[499]
for i in range(len(accuracy_psogwo_4)):
    if accuracy_psogwo_4[i] > 0:
        accuracy_psogwo_4[i] = np.log(accuracy_psogwo_4[i]) / np.log(10)
    else:
        accuracy_psogwo_4[i] = -18
# PSO-GWO优化
# PSO-GWO参数


print('---------------- 寻优结果 -----------------')

plot(iterations_gwo_1, iterations_Igwo_1, iterations_woa_1, iterations_psogwo_1, accuracy_gwo_1, accuracy_Igwo_1,
     accuracy_woa_1, accuracy_psogwo_1,
     iterations_gwo_2, iterations_Igwo_2, iterations_woa_2, iterations_psogwo_2, accuracy_gwo_2, accuracy_Igwo_2,
     accuracy_woa_2, accuracy_psogwo_2,
     iterations_gwo_3, iterations_Igwo_3, iterations_woa_3, iterations_psogwo_3, accuracy_gwo_3, accuracy_Igwo_3,
     accuracy_woa_3, accuracy_psogwo_3,
     iterations_gwo_4, iterations_Igwo_4, iterations_woa_4, iterations_psogwo_4, accuracy_gwo_4, accuracy_Igwo_4,
     accuracy_woa_4, accuracy_psogwo_4
     )

# #统计平均值
# #函数F1
# import csv
# with open('testF1_FPSOGWO.csv', 'r') as file:
#     reader = csv.reader(file)
#     data_1 = list(reader)
# new_data_1 = [GWO_best_1,PSO_best_1,WOA_best_1,PSOGWO_best_1]
# data_1.append(new_data_1)
#
# with open('testF1_FPSOGWO.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(data_1)
#
# #函数F2
# with open('testF2_FPSOGWO.csv', 'r') as file:
#     reader = csv.reader(file)
#     data_2 = list(reader)
# new_data_2 = [GWO_best_2,PSO_best_2,WOA_best_2,PSOGWO_best_2]
# data_2.append(new_data_2)
#
# with open('testF2_FPSOGWO.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(data_2)
#
# #函数F3
# with open('testF3_FPSOGWO.csv', 'r') as file:
#     reader = csv.reader(file)
#     data_3 = list(reader)
# new_data_3 = [GWO_best_3,PSO_best_3,WOA_best_3,PSOGWO_best_3]
# data_3.append(new_data_3)
#
# with open('testF3_FPSOGWO.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(data_3)
#
#
# #函数F4
# with open('testF4_FPSOGWO.csv', 'r') as file:
#     reader = csv.reader(file)
#     data_4 = list(reader)
# new_data_4 = [GWO_best_4,PSO_best_4,WOA_best_4,PSOGWO_best_4]
# data_4.append(new_data_4)
#
# with open('testF4_FPSOGWO.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(data_4)
#

'''
SVR模型
'''
regressor.fit(X_train_s, y_train)  # 拟合SVR模型

y_pred_svr = regressor.predict(X_valid_s)

# APPLYING K-FOLD CROSS VALIDATION on SVR model
accuracies = cross_val_score(regressor, X=X_test_s, y=y_test, cv=10, scoring='neg_mean_squared_error')
accuracy_mean = accuracies.mean()
accuracies.std() * 100

mse = mean_squared_error(y_valid, y_pred_svr)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, y_pred_svr)
nrmse = rmse / (y_valid.max() - y_valid.min())
print('未优化SVR模型分数：', regressor.score(X_test_s, y_test))  # 分数
print("未优化的SVR模型评价结果：")
print("RMSE =", rmse)
print("MSE =", mse)
print("Normalized RMSE=", nrmse)
print("R Square =", r2)
print("K-fold accuracy mean", accuracy_mean)

# 创建一个SVR验证集的图形
plt.figure(figsize=(10, 6))

# 绘制真实值的折线
plt.plot(y_valid, label='真实值', marker='o', linestyle='-')

# 绘制预测值的折线
plt.plot(y_pred_svr, label='预测值', marker='x', linestyle='-')
plt.title('未优化SVR模型下验证集真实值与预测值')
plt.show()
