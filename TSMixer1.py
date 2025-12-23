# 导入库（复用你原代码的库，新增TSMixer核心层定义）
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

start_time = time.time()

# -------------------------- 1. 数据加载与预处理（完全复用你的原代码）--------------------------
# 特征标准化（复用）
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# 加载数据集（复用你的路径）
train_dataSet = pd.read_csv(
    r"D:\linshiwenjian\机器学习\MLFinalCode\Dataset\加噪数据集\modified_数据集Time_Series661_detail.dat")
test_dataSet = pd.read_csv(
    r"D:\linshiwenjian\机器学习\MLFinalCode\Dataset\加噪数据集\modified_数据集Time_Series662_detail.dat")

# 定义列名（复用你的columns/noise_columns）
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
CL = columns + noise_columns

# 缺失值处理（复用你的前向填充）
train_dataSet = train_dataSet.ffill()
test_dataSet = test_dataSet.ffill()

# 异常值处理（复用你的Z分数法，可选）
for column in CL:
    z_scores = np.abs(stats.zscore(train_dataSet[column]))
    outlier_idx = z_scores > 2
    train_dataSet.loc[outlier_idx, column] = train_dataSet[column].median()
    z_scores_test = np.abs(stats.zscore(test_dataSet[column]))
    outlier_idx_test = z_scores_test > 2
    test_dataSet.loc[outlier_idx_test, column] = train_dataSet[column].median()

# 划分X/y（复用，输入是noise_columns，输出是columns）
X_train = train_dataSet[noise_columns].values  # (样本数, 6) 2D格式
y_train = train_dataSet[columns].values  # (样本数, 6)
X_test = test_dataSet[noise_columns].values
y_test = test_dataSet[columns].values

# 标准化（复用）
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)


# -------------------------- 2. TSMixer核心层定义（新增，适配你的6特征时序数据）--------------------------
class MixerLayer(Layer):
    """TSMixer的核心层：包含Time Mixing和Feature Mixing"""

    def __init__(self, time_steps, hidden_dim=32, dropout_rate=0.2):
        super(MixerLayer, self).__init__()
        self.hidden_dim = hidden_dim  # MLP隐藏层维度
        self.dropout = Dropout(dropout_rate)  # 防止过拟合（适配你的噪声数据）

        # 1. Time Mixing：捕捉单个噪声特征的时序依赖（如Error_T_SONIC的时间趋势）
        self.time_mlp = Sequential([
            Dense(hidden_dim, activation='relu'),
            Dropout(dropout_rate),
            Dense(time_steps)  # 改成“时间步数量”（比如你的5）
        ])

        # 2. Feature Mixing：捕捉同一时间步下6个噪声特征的关联（如Error_CO2_density与Error_CO2_sig_strgth）
        self.feature_mlp = Sequential([
            Dense(hidden_dim, activation='relu'),  # 非线性激活，拟合特征关联
            Dropout(dropout_rate),
            Dense(X_train_scaled.shape[1])  # 输出维度=特征数（6），跨特征共享权重
        ])

        # 归一化层（稳定训练，适配传感器数据的数值波动）
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # inputs: (batch_size, 时间步, 6) → 时序格式的噪声特征数据
        batch_size, time_steps, n_features = inputs.shape

        # Step1: Time Mixing（对每个特征的时间序列单独处理）
        x_time = self.layer_norm(inputs)  # 归一化
        x_time = tf.transpose(x_time, perm=[0, 2, 1])  # (batch_size, 6, 时间步) → 转置为“特征×时间”
        x_time = self.time_mlp(x_time)  # MLP处理时序
        x_time = tf.transpose(x_time, perm=[0, 2, 1])  # 转置回（batch_size, 时间步, 6）
        x_time = self.dropout(x_time)
        x = inputs + x_time  # 残差连接：保留原始噪声特征信息

        # Step2: Feature Mixing（对每个时间步的6个特征单独处理）
        x_feature = self.layer_norm(x)  # 归一化
        x_feature = self.feature_mlp(x_feature)  # MLP处理特征关联
        x_feature = self.dropout(x_feature)
        x = x + x_feature  # 残差连接：保留时序处理后的信息

        return x


# -------------------------- 3. 构建TSMixer模型（替换原LSTM模型）--------------------------
# 配置参数（适配你的6特征数据，简化易调参）
time_steps = 5  # 时间步（和你原LSTM一致，用过去5个时间步预测当前原始值）
hidden_dim = 32  # MLP隐藏层维度
dropout_rate = 0.2
num_mixer_layers = 2  # Mixer Layer层数（不用多，2-3层足够）


# 数据格式转换：从2D→3D（适配MixerLayer的时序输入，和你原LSTM的create_sequences逻辑一致）
def create_sequences(X, y, time_steps):
    X_seq, y_seq = [], []
    for i in range(time_steps, len(X)):
        X_seq.append(X[i - time_steps:i, :])  # (5, 6) → 过去5个时间步的6个噪声特征
        y_seq.append(y[i, :])  # 当前时间步的6个原始特征（目标值）
    return np.array(X_seq), np.array(y_seq)


X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)  # (N, 5, 6)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)  # (M, 5, 6)

# 构建TSMixer模型
model = Sequential()
# 输入层：(时间步=5, 特征数=6) → 完全对齐你的噪声特征数
model.add(tf.keras.Input(shape=(time_steps, X_train_scaled.shape[1])))
# 添加2个Mixer Layer（捕捉时序+特征信息）
for _ in range(num_mixer_layers):
    model.add(MixerLayer(time_steps=time_steps, hidden_dim=hidden_dim, dropout_rate=dropout_rate))  # 传入 time_steps
# 输出层：预测6个原始特征值（和你原LSTM的输出维度一致）
model.add(Dense(6))

# 编译模型（复用你的优化器和损失函数，保持一致性）
model.compile(optimizer='adam', loss='mse')

# 早停策略（复用，防止过拟合）
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# -------------------------- 4. 训练、预测、评估（完全复用你的原代码）--------------------------
# 模型训练
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 预测
y_predict_scaled = model.predict(X_test_seq)
y_predict = scaler_y.inverse_transform(y_predict_scaled)  # 反标准化
y_test_actual = scaler_y.inverse_transform(y_test_seq)


# 模型评估（复用你的metrics_sklearn函数）
def metrics_sklearn(y_valid, y_pred_):
    r2 = r2_score(y_valid, y_pred_)
    print('r2_score:{0}'.format(r2))
    mse = mean_squared_error(y_valid, y_pred_)
    print('mse:{0}'.format(mse))


metrics_sklearn(y_test_actual, y_predict)

# 结果保存（复用）
results = []
for True_Value, Predicted_Value in zip(y_test_actual, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_TSMixer.csv", index=False)

# 误差分析（复用）
print("<*>" * 50)
data = pd.read_csv("result_TSMixer.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()
print("6个数据的平均值为：\n", means)
print(means.mean())

end_time = time.time()
print(f"总耗时：{end_time - start_time : .3f}秒")