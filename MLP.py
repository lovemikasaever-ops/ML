# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main_mlp.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor  # 导入MLP回归器
from scipy import stats

start_time = time.time()

# 特征标准化（MLP对特征缩放更敏感，这里实际应用标准化）
scaler = StandardScaler()

# 加载数据集
train_dataSet = pd.read_csv(r"D:\linshiwenjian\机器学习\MLFinalCode\Dataset\加噪数据集\modified_数据集Time_Series661_detail.dat")
test_dataSet = pd.read_csv(r"D:\linshiwenjian\机器学习\MLFinalCode\Dataset\加噪数据集\modified_数据集Time_Series662_detail.dat")

# 定义列名
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
CL = columns + noise_columns

# 查看数据缺失情况
data = train_dataSet[CL]
missingDf = data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns = ['feature', 'miss_num']
missingDf['miss_percentage'] = missingDf['miss_num'] / data.shape[0]  # 缺失值比例
print("缺失值比例")
print(missingDf)

# 计算异常值比例
outlier_ratios = {}
for column in CL:
    z_scores = np.abs(stats.zscore(train_dataSet[column]))
    outliers = (z_scores > 2)
    outlier_ratio = outliers.mean()
    outlier_ratios[column] = outlier_ratio
print("*" * 30)
print("异常值的比例:")
for column, ratio in outlier_ratios.items():
    print(f"{column}: {ratio:.2%}")

# 划分训练集和测试集
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# 对特征进行标准化（MLP必须做特征缩放）
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""MLP模型参数设置"""
# 网格搜索参数（注释部分为可调整的参数范围）
params = {
    # 'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
    # 'activation': ['relu', 'tanh'],
    # 'solver': ['adam', 'sgd'],
    # 'learning_rate': ['constant', 'adaptive'],
    # 'max_iter': [200, 300, 500]
}

# MLP模型基础参数
other_params = {
    'hidden_layer_sizes': (128, 64),  # 两层隐藏层，分别128和64个神经元
    'activation': 'relu',             # 激活函数
    'solver': 'adam',                 # 优化器
    'learning_rate': 'adaptive',      # 学习率调整方式
    'max_iter': 300,                  # 最大迭代次数
    'random_state': 217,              # 随机种子，保证结果可复现
    'verbose': False                  # 训练时不打印详细信息
}

# 初始化MLP回归模型
model_adj = MLPRegressor(** other_params)

# # 网格搜索调参（如需调参可取消注释）
# optimized_param = GridSearchCV(
#     estimator=model_adj,
#     param_grid=params,
#     scoring='r2',
#     cv=5,
#     verbose=1
# )
# optimized_param.fit(X_train_scaled, y_train)
# print('参数的最佳取值：{0}'.format(optimized_param.best_params_))
# print('最佳模型得分:{0}'.format(optimized_param.best_score_))
# model_adj = optimized_param.best_estimator_  # 使用最佳参数模型

# 模型训练
model_adj.fit(X_train_scaled, y_train)

# 预测值
y_predict = model_adj.predict(X_test_scaled)

# 模型效果评估函数
def metrics_sklearn(y_valid, y_pred_):
    r2 = r2_score(y_valid, y_pred_)
    print('r2_score:{0}'.format(r2))
    mse = mean_squared_error(y_valid, y_pred_)
    print('mse:{0}'.format(mse))

# 评估模型
metrics_sklearn(y_test, y_predict)

# 结果处理与保存
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_MLP1.csv", index=False)

print("<*>" * 50)

# 计算误差平均值
data = pd.read_csv("result_MLP.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()
print("6个数据的平均值为：\n", means)
print(means.mean())

end_time = time.time()
print(f"总耗时：{end_time - start_time : .3f}秒")