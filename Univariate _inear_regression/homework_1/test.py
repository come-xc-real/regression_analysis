# 导入相关模块
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 设置显示中文
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

#读取数据
df = pd.read_csv('我国1978~1997年的财政收入y和国民生产总值x.csv')


X = np.array(df['x']).reshape(-1, 1)
Y = np.array(df['y']).reshape(-1, 1)
plt.scatter(X,Y)


# 模型搭建
model = LinearRegression()
model.fit(X,Y)

# 模型可视化
plt.scatter(X,Y)
plt.plot(X,model.predict(X),color='red')
plt.xlabel('工龄')
plt.ylabel('薪水')
plt.show()

import statsmodels.api as sm

# add_constant()函数给原来的特征变量X添加常数项，并赋给X2，这样才有y＝ax＋b中的常数项，即截距b
X2 = sm.add_constant(X)
# 用OLS()和fit()函数对Y和X2进行线性回归方程搭建
est = sm.OLS(Y,X).fit()
print(est.summary())
