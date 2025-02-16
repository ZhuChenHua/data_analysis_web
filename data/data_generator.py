import pandas as pd
import numpy as np


# 随机生成一组数据，输出为CSV文件
# 定义一个函数，用于生成指定数量的样本和特征的数据，并将其保存为CSV文件
def generate_data(num_samples, num_features):
    # 使用numpy的random.rand函数生成一个二维数组，数组的行数为num_samples，列数为num_features
    # 数组中的元素是0到10之间的整数
    data = np.random.randint(0, 10, size=(num_samples, num_features))
    # 将生成的二维数组转换为pandas的DataFrame对象
    df = pd.DataFrame(data)

    # 存放路径
    path = "data/"

    # 将DataFrame对象保存为CSV文件，文件名为data1.csv，不保存行索引
    df.to_csv(path + "data1.csv", index=False)


if __name__ == "__main__":
    generate_data(100, 1)  # 生成1000个样本，10个特征
