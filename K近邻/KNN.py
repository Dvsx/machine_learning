from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
# 加载数据
digits = load_digits()
data = digits.data
# 输出数据的维度
print(data.shape)
# 输出第一个数据的数组值
print(digits.images[0])
# 输出第一个数据的标签
print(digits.target[0])
# 画出第一张图
plt.gray()
plt.imshow(digits.images[0])
plt.show()
#分割数据集
train_x,test_x,train_y,test_y = train_test_split(data,digits.target,test_size=0.25,random_state=33)
# 采用z-score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)
# 创建KNN分类器
knn = KNeighborsClassifier()
knn.fit(train_ss_x,train_y)
predict_y = knn.predict(test_ss_x)
print("KNN 准确率：%.4lf"%accuracy_score(predict_y,test_y))

# 创建SVM分类器
svm = SVC()
svm.fit(train_ss_x,train_y)
predict_y = svm.predict(test_ss_x)
print('SVM准确率：%.4lf'%accuracy_score(predict_y,test_y))

# min-max规范化
mm = preprocessing.MinMaxScaler()
train_mm_x = mm.fit_transform(train_x)
test_mm_x = mm.transform(test_x)

# 创建CART决策树分类器
dtc = DecisionTreeClassifier()
dtc.fit(train_mm_x,train_y)
predict_y = dtc.predict(test_mm_x)
print("CART决策树准确率：%.4lf"%accuracy_score(predict_y,test_y))

# 创建贝叶斯分类器
mnb = MultinomialNB()
mnb.fit(train_mm_x,train_y)
predict_y = mnb.predict(test_mm_x)
print(" 多项式朴素贝叶斯准确率: %.4lf" % accuracy_score(predict_y, test_y))
