import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

data = pd.read_csv("t3.3.csv", sep=" ")
data = np.array(data)


# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:,1], data[:,2], data[:,0])


# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('x_1', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('x_2', fontdict={'size': 15, 'color': 'red'})

plt.show()