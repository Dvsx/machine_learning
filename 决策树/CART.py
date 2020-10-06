# CART分类树
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
# 准备数据集
iris = load_iris()
# 获取特征集和分类标识
features = iris.data
labels = iris.target
# 划分特征和标签
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)
# 创建CART分类树
clf = DecisionTreeClassifier(criterion='gini')
# 拟合构造CART分类树
clf = clf.fit(train_features,train_labels)
# 预测
test_predict = clf.predict(test_features)
# 对比
score = accuracy_score(test_labels,test_predict)
print("CART分类树准确率%.4lf" % score)

# CART回归树
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor

boston = load_boston()
print(boston.feature_names)
features = boston.data
prices = boston.target
train_features, test_features, train_price, test_price = train_test_split(features, prices, test_size=0.33)

dtr = DecisionTreeRegressor()
dtr.fit(train_features,train_price)
predict_price = dtr.predict(test_features)
print(predict_price)
print('回归树二乘偏差均值:', mean_squared_error(test_price, predict_price))
print('回归树绝对值偏差均值:', mean_absolute_error(test_price, predict_price)) 