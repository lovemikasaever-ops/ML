# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main_lstm.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

start_time = time.time()

# 特征标准化
scaler_X = StandardScaler()  # 用于输入特征的标准化
scaler_y = StandardScaler()  # 用于目标值的标准化

# 加载数据集
train_dataSet = pd.read_csv(r"D:\linshiwenjian\机器学习\MLFinalCode\Dataset\数据集（含真实值）\modified_数据集Time_Series661.dat")
test_dataSet = pd.read_csv(r"D:\linshiwenjian\机器学习\MLFinalCode\Dataset\数据集（含真实值）\modified_数据集Time_Series662.dat")

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
    z_scores = np.abs(stats.zscore(train_dataSet[column].dropna()))  # 忽略NaN值计算Z分数
    outliers = (z_scores > 2)
    outlier_ratio = outliers.mean()
    outlier_ratios[column] = outlier_ratio
print("*" * 30)
print("异常值的比例:")
for column, ratio in outlier_ratios.items():
    print(f"{column}: {ratio:.2%}")

# 处理缺失值（LSTM不能处理NaN，这里使用前向填充）
train_dataSet = train_dataSet.fillna(method='ffill')
test_dataSet = test_dataSet.fillna(method='ffill')

# 划分训练集和测试集
X_train = train_dataSet[noise_columns].values
y_train = train_dataSet[columns].values
X_test = test_dataSet[noise_columns].values
y_test = test_dataSet[columns].values

# 数据标准化
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 转换为LSTM需要的3D格式 [样本数, 时间步, 特征数]
# 这里使用时间步为5（可根据实际数据序列特性调整）
time_steps = 5

def create_sequences(X, y, time_steps):
    """将数据转换为序列数据"""
    X_seq, y_seq = [], []
    for i in range(time_steps, len(X)):
        X_seq.append(X[i-time_steps:i, :])  # 取前time_steps行作为输入序列
        y_seq.append(y[i, :])  # 取第i行作为目标值
    return np.array(X_seq), np.array(y_seq)

# 创建序列数据
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

# 构建LSTM模型
model = Sequential()
# 第一层LSTM，64个神经元，返回序列（因为下一层还是LSTM）
model.add(LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=True))
model.add(Dropout(0.2))  # 防止过拟合
# 第二层LSTM，32个神经元
model.add(LSTM(32))
model.add(Dropout(0.2))
# 输出层，预测6个特征
model.add(Dense(6))

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 早停策略（防止过拟合）
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 模型训练
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.2,  # 用20%训练数据作为验证集
    callbacks=[early_stopping],
    verbose=1
)

# 预测
y_predict_scaled = model.predict(X_test_seq)
# 将预测值反标准化（恢复原始尺度）
y_predict = scaler_y.inverse_transform(y_predict_scaled)
# 对应的真实值也需要调整（因为序列数据舍弃了前time_steps行）
y_test_actual = scaler_y.inverse_transform(y_test_seq)

# 模型效果评估函数
def metrics_sklearn(y_valid, y_pred_):
    r2 = r2_score(y_valid, y_pred_)
    print('r2_score:{0}'.format(r2))
    mse = mean_squared_error(y_valid, y_pred_)
    print('mse:{0}'.format(mse))

# 评估模型
metrics_sklearn(y_test_actual, y_predict)

# 结果处理与保存
results = []
for True_Value, Predicted_Value in zip(y_test_actual, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_LSTM.csv", index=False)

print("<*>" * 50)

# 计算误差平均值
data = pd.read_csv("result_LSTM.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()
print("6个数据的平均值为：\n", means)
print(means.mean())

end_time = time.time()
print(f"总耗时：{end_time - start_time : .3f}秒")