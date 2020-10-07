import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
# 远程读取文件
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)

X_train=digits_train[np.arange(64)]
y_train=digits_train[64]
 
X_test=digits_test[np.arange(64)]
y_test=digits_test[64]
print(y_test)
# kmeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)
print(metrics.adjusted_rand_score(y_test,y_pred))

