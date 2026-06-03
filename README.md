# project
some projects that i worked on during my college years

## project 1：基于CHARLS数据的老年人健康模式识别
- 数据来源：CHARLS 2020全国调查数据
- 模块1：数据清洗预处理：[data_processing.ipynb](./charls_health_project/data_processing.ipynb)
- 模块2：聚类建模与可视化：[clustering and visualization.ipynb](./charls_health_project/clustering&visualization.ipynb)
- 实现方法：数据预处理、PCA降维、Kmeans聚类、t-SNE/雷达图可视化，划分三类老年健康群体

## 项目2：基于DeepLOB的高频股票Tick数据预测与结果可视化
- 项目简介：本项目基于 DeepLOB（Deep Convolutional Neural Networks for Limit Order Books）架构，结合双向 LSTM（BiLSTM）实现了基于订单簿数据的多任务价格方向预测系统。该系统以金融市场的订单簿深度数据（买卖盘各 5 档价格与成交量）为输入，通过构建多维度衍生特征、深度卷积特征提取、双向时序建模，最终实现对不同时间跨度下中间价移动方向（上涨 / 横盘 / 下跌）的多任务预测，可应用于算法交易、市场流动性分析等金融场景。
- 环境配置：[requirements.txt](./deeplob/requirements.txt)
- 模型结构代码：[model.py](./deeplob/model.py)
- 推理预测代码：[Predictor.py](./deeplob/Predictor.py)
- 配置参数：[config.json](./deeplob/config.json)
- model.pth 请自己做train.py以获得model.pth进行预测



## 技术栈
Python | Pandas | Matplotlib | Seaborn | Scikit-learn | Tableau