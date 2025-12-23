# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy import stats

start_time = time.time()

# 特征标准化 StandardScaler是 scikit-learn 库中用于数据标准化的工具，其核心功能是将数据转换为均值为 0、标准差为 1的标准正态分布形式
#scaler = StandardScaler()

# 加载数据集 训练时，从训练集train_dataSet中提取columns对应的列作为模型的输出（y_train），即模型需要学习预测的原始特征值；
# 测试时，从测试集test_dataSet中提取columns对应的列作为真实标签（y_test），用于评估模型预测效果。
train_dataSet = pd.read_csv(r'D:\linshiwenjian\机器学习\MLFinalCode\Dataset\加噪数据集\modified_数据集Time_Series661_detail.dat')

test_dataSet = pd.read_csv(r'D:\linshiwenjian\机器学习\MLFinalCode\Dataset\加噪数据集\modified_数据集Time_Series662_detail.dat')


# columns表示原始列 数据集中原始特征列的列名 ，noise_columns表示添加噪声的额列 noise_columns列表的核心作用是标识模型的输入特征
#从训练集train_dataSet中提取noise_columns对应的列作为模型的输入（X_train）；
#从测试集test_dataSet中提取noise_columns对应的列作为模型的测试输入（X_test）。
#结合整体代码逻辑，这些带噪声的特征是模型的 “输入”，用于预测原始特征的真实值（即columns对应的列作为模型的输出目标y_train/y_test），本质是通过机器学习模型学习 “噪声特征” 与 “原始特征” 之间的映射关系，实现从噪声数据中恢复原始特征的目的。
#columns 中的特征（如 T_SONIC、CO2_density 等）是原始的、准确的特征值（可理解为 “真实值”）；
#noise_columns 中的特征（如 Error_T_SONIC、Error_CO2_density 等）是在原始特征的基础上，通过添加随机误差、测量噪声或其他干扰因素得到的带有偏差的特征值（可理解为 “受噪声污染的观测值”）。
#这种设计的目的是训练模型学习 “带噪声的观测值” 与 “原始真实值” 之间的映射关系，最终实现从噪声数据中恢复或预测原始真实特征值的功能（例如模拟实际场景中传感器测量存在噪声时，通过模型修正噪声以获得更准确的数据）。
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth',]
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
#CL包含原始特征和带噪声的特征，共 12 列
CL = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth','Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

## 查看数据缺失情况
#data = train_dataSet[CL]从训练数据集（train_dataSet）中提取CL列表包含的所有列（CL是原始特征列columns和带噪声特征列noise_columns的组合），形成一个新的 DataFrame data，用于后续缺失值分析。
#missingDf = data.isnull().sum().sort_values(ascending=False).reset_index()
#data.isnull()：生成与data结构相同的布尔型 DataFrame，True表示对应位置为缺失值（NaN）。
#.sum()：对每一列的缺失值进行计数（True计为 1，False计为 0），得到一个 Series，索引为特征名，值为该特征的缺失值总数。
#.sort_values(ascending=False)：按缺失值数量从多到少对 Series 排序。
##.reset_index()：将排序后的 Series 转换为 DataFrame，原索引（特征名）变为新列index，缺失值数量为另一列。
#missingDf.columns = ['feature', 'miss_num']为missingDf的列重命名：原index列（特征名）改为feature，原缺失值数量列改为miss_num（表示缺失值数量）。
#missingDf['miss_percentage'] = missingDf['miss_num'] / data.shape[0]计算每个特征的缺失值比例：用缺失值数量（miss_num）除以训练集总样本数（data.shape[0]，即行数），结果作为新列miss_percentage添加到missingDf中。
data = train_dataSet[CL]
missingDf=data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns=['feature','miss_num']
missingDf['miss_percentage']=missingDf['miss_num']/data.shape[0]  #缺失值比例
print("缺失值比例")
print(missingDf)


# 初始化一个字典来存储每一列的异常值比例 功能是分析训练数据集中各特征的异常值比例，属于数据预处理中的数据质量评估环节
outlier_ratios = {} #初始化一个空字典，用于存储每个特征（列）对应的异常值比例，键为特征名，值为该特征的异常值比例。

# 遍历每一列
for column in CL: #遍历CL列表中的每一个特征列（CL包含原始特征和带噪声的特征，共 12 列），对每一列单独进行异常值分析。
    # 计算每一列的Z分数 （标准化分数）用于衡量数据点偏离均值的程度（以标准差为单位
    z_scores = np.abs(stats.zscore(train_dataSet[column]))

    # 找出异常值（假设Z分数大于2为异常值） 当 Z 分数的绝对值大于 2 时，认为该数据点是异常值。这是一个经验阈值（Z 分数的绝对值越大，数据点越异常，此处选择 2 作为判断标准），结果是一个布尔数组（True表示异常值，False表示正常数据）。
    outliers = (z_scores > 2)

    # 计算异常值的比例 计算该列的异常值比例：布尔数组中True的数量占总数据量的比例（True在计算均值时等价于 1，False等价于 0，因此均值即异常值占比）。
    outlier_ratio = outliers.mean()

    # 存储异常值比例 将当前列的异常值比例存入字典outlier_ratios，键为列名，值为比例。
    outlier_ratios[column] = outlier_ratio
print("*"*30)
# 打印结果
print("异常值的比例:")
for column, ratio in outlier_ratios.items():
    print(f"{column}: {ratio:.2%}")



# 划分训练集中X_Train和y_Train
#y_train：训练集的输出目标，取自训练数据中columns对应的列（即原始无噪声的特征，共 6 列），是模型需要学习预测的 “真实值”。
#X_test：测试集的输入特征，取自测试数据（test_dataSet）中noise_columns对应的列，用于模型测试时的输入。
#y_test：测试集的输出目标，取自测试数据中columns对应的列，作为模型预测结果的 “真实标签”，用于评估模型效果。
X_train = train_dataSet[noise_columns]

y_train = train_dataSet[columns]

# 划分测试集中X_test和y_test
X_test = test_dataSet[noise_columns]

y_test = test_dataSet[columns]

"""模型调参"""
params = {
    # 'n_estimators': [120, 150, 180, 200],
    # 'learning_rate': [0.1, 0.15, 0.2],
    # 'max_depth': [2, 3, 4, 5],
    # "reg_alpha": [8, 10, 20, 30],
    # "reg_lambda": [6, 12],
    # "min_child_weight": [2, 3, 4, 5, 6],
    # 'subsample': [i / 100.0 for i in range(60, 80, 5)],
    # 'colsample_bytree': [i / 100.0 for i in range(80, 100, 5)]
}
#seed: 217随机种子，用于控制模型训练过程中的随机性（如样本 / 特征采样、树的初始化等）。固定种子可确保实验结果可复现，多次运行时初始条件一致。
#booster: 'gbtree'指定基学习器类型。gbtree表示使用基于决策树的模型作为弱学习器（XGBoost 的核心），另一种可选值为gblinear（线性模型）。此处选择树模型，因其对非线性关系的拟合能力更强。
#max_depth: 5每棵决策树的最大深度。深度越大，树的结构越复杂（可捕捉更多细节），但也更容易过拟合。设置为 5 是平衡模型复杂度和过拟合风险的折中选择。
#n_estimators: 200集成模型中决策树的数量（即迭代次数）。树的数量越多，模型的拟合能力越强，但计算成本也越高。200 是一个适中的数量，兼顾性能和效率。
#learning_rate: 0.1学习率（又称 “步长收缩”），用于缩放每棵树的预测贡献。较小的学习率可降低过拟合风险，但通常需要配合更多的n_estimators（树的数量）才能达到较好效果。此处 0.1 是常用的基准值。
#gamma: 0节点分裂的最小损失减少阈值。当分裂节点带来的损失减少量大于gamma时，才允许分裂。gamma=0表示不限制分裂条件，允许模型尽可能分裂以拟合数据。
#reg_alpha: 10L1 正则化参数，用于惩罚树的叶子节点权重。值越大，正则化强度越高，可抑制过拟合（通过强制部分权重为 0，实现特征选择效果）。此处 10 表示较强的 L1 约束。
#reg_lambda: 6L2 正则化参数，同样用于惩罚叶子节点权重，但 L2 倾向于让权重值更分散（而非强制为 0）。6 是中等强度的 L2 约束，辅助控制过拟合。
#min_child_weight: 5子节点的最小样本权重和。当某节点分裂后，子节点的样本权重和需大于等于该值才允许分裂。值越大，模型越保守（避免为少量样本分裂），此处 5 是平衡保守性的设置。
#colsample_bytree: 0.85每棵树训练时随机选择的特征比例（此处为 85%）。通过随机采样特征，减少特征间的相关性，降低过拟合风险，同时加速训练。
#subsample: 0.6每棵树训练时随机选择的样本比例（此处为 60%）。通过样本采样增加随机性，避免模型过度依赖某些样本，进一步抑制过拟合。
other_params = {
    'seed': 217,
    'booster': 'gbtree',
    'max_depth': 5,
    'n_estimators': 200,
    'learning_rate': 0.1,
    'gamma': 0,
    'reg_alpha': 10,
    'reg_lambda': 6,
    'min_child_weight': 5,
    'colsample_bytree': 0.85,
    'subsample': 0.6,
}

model_adj = XGBRegressor(**other_params)

# # sklearn提供的调参工具，训练集k折交叉验证(消除数据切分产生数据分布不均匀的影响)
# optimized_param = GridSearchCV(estimator=model_adj, param_grid=params, scoring='r2', cv=5, verbose=1)
# # 模型训练
# optimized_param.fit(X_train, y_train)
#
# # 对应参数的k折交叉验证平均得分
# means = optimized_param.cv_results_['mean_test_score']
# params = optimized_param.cv_results_['params']
# for mean, param in zip(means, params):
#     print("mean_score: %f,  params: %r" % (mean, param))
# # 最佳模型参数
# print('参数的最佳取值：{0}'.format(optimized_param.best_params_))
# # 最佳参数模型得分
# print('最佳模型得分:{0}'.format(optimized_param.best_score_))


# 模型训练
model_adj.fit(X_train, y_train)

# # 模型保存
# model_adj.save_model('xgb_regressor.json')
#
# # 模型加载
# model_adj = XGBRegressor()
# model_adj.load_model('xgb_regressor.json')

# 预测值
y_predict = model_adj.predict(X_test)


# def metrics_sklearn(y_valid, y_pred_):
#     """模型效果评估"""
#     r2 = r2_score(y_valid, y_pred_)
#     print('r2_score:{0}'.format(r2))
#
#     mse = mean_squared_error(y_valid, y_pred_)
#     print('mse:{0}'.format(mse))
#
#
#
# """模型效果评估"""
# metrics_sklearn(y_test, y_predict)

results = []
# 遍历y_test和y_predict，并且计算误差
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    # 格式化True_Value和Predicted_Value为原始数据格式
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))  # 修改ERROR数据格式
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])  # 保存结果


# 结果写入CSV文件当中 作用：处理预测结果并保存到 CSV 文件。
# 遍历测试集真实值（y_test）和预测值（y_predict），计算每个特征的绝对误差（np.abs(...)）。
# 将真实值、预测值、误差格式化为字符串（便于后续读取），存入列表results。
# 转换为 DataFrame 并写入result_XGB.csv，持久化保存结果。
result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_XGB1.csv", index=False)

print("<*>"*50)

# 从CSV文件读取数据
data = pd.read_csv("result_XGB1.csv")

# 提取第三列数据
column3 = data.iloc[:, 2]

# 将每行的7个数字拆分并转换为数字列表
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)

# 计算平均值
means = numbers.mean()

# 打印结果
print("6个数据的平均值为：\n", means)
print(means.mean())

end_time = time.time()
print(f"总耗时：{end_time - start_time : .3f}秒")