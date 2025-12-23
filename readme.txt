文件说明：
- 数据集(真实值)Time_Series_661.dat ：训练集数据
- 数据集（真实值）Time_Series_662.dat：预测集数据

运行说明：
- step01：将本项目中所有的文件，移入一个纯净的python项目中
- step02：执行add_noise.py文件，随机添加噪声
【执行完，在项目当前目录下，生成modified_数据集Time_Series661_detail.dat与modified_数据集Time_Series662_detail.dat】
- step03：执行XGBRegressor文件，训练模型并预测数据
-step04：将最后输出的预测数据csv文件和原始含真实值的 数据集（真实值）Time_Series_662.dat放入比较代码evaluate_predictions.py执行，输出最后误差值。不断改进模型，最后的误差值越小越好。


环境说明：
- python==3.9
- scikit-learn==1.4.1.post1  
- pandas==2.1.2
- numpy==1.26.1
- scipy==1.12.0

比赛说明：
-大家需要修改代码中定义训练模型部分，可以选择更好的模型