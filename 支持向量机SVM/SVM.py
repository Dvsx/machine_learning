# encoding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("D:/DATA/breast_cancer_data-master/data.csv",encoding='utf-8') #这里要注意，如果文件中有中文，本地文件一定要转换成 UTF-8的编码格式
# 数据探索
# 因为数据集中列比较多，我们需要把 dataframe 中的列全部显示出来
pd.set_option('display.max_columns', None)
#print(data.columns)
#print(data.head(5))
#print(data.describe())
# 将 B 良性替换为 0，M 恶性替换为 1
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})# encoding=utf-8

#数据清洗
# 将特征字段分成 3 组
features_mean= list(data.columns[2:12])
features_se= list(data.columns[12:22])
features_worst=list(data.columns[22:32])
# 数据清洗
# ID 列没有用，删除该列
#data.drop(columns=['id'],axis=1,inplace=True)

# 用热力图呈现 features_mean 字段之间的相关性
# 计算列与列之间的相关系数
corr = data[features_mean].corr()
plt.figure(figsize=(8,8))
# annot=True 显示每个方格的数据
sns.heatmap(corr, annot=True)
plt.show()

# 特征选择，只选择了平均值这一维度，并且还去除了相关系数较大的特征值
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean'] 
# 分割数据集，分成测试集和训练集
train,test = train_test_split(data,test_size = 0.3)
train_X = train[features_remain]
train_Y = train['diagnosis']
test_X = test[features_remain]
test_Y = test['diagnosis']
# 采用Z-Score进行标准化，公式 X-mean/std
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)
# 创建分类器，使用SVC核函数
model = svm.SVC()
model.fit(train_X,train_Y)
prediction = model.predict(test_X)
print('准确率',metrics.accuracy_score(prediction,test_Y))


# https://zhuanlan.zhihu.com/p/66235389