# 十二生肖数据集

根据星河社区十二生肖图片数据集（https://aistudio.baidu.com/datasetdetail/171752）
，生成Numpy格式原始数据集，并通过百度飞桨框架PaddlePaddle，封装为PaddleVision格式数据集。

### 数据集使用

1. 安装PaddlePaddle 3.0以上版本
2. 加载数据集
```python
from shengxiao import SHENGXIAO

ds = SHENGXIAO()
```