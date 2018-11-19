
from sklearn.cross_validation  import train_test_split
from sklearn import metrics, neighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from graphviz import Source
from IPython.display import SVG  


df2=pd.read_csv("restaurant_30.csv",encoding="big5")
df3=df2[['AC','delicious','decorate','place','service','popular']]

#切分訓練 測試資料
x=df3[['AC','delicious','decorate','place','service']]
y=df3[['popular']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) 
#test_size=0.2  =>  假設他學 8 成, 能準確預估出剩餘2成的幾成?

# 選擇 k
range = np.arange(1, round(0.2 * x_train.shape[0]) + 1)
accuracies = []

for i in range:
    clf = neighbors.KNeighborsClassifier(n_neighbors = i)
    iris_clf = clf.fit(x_train, y_train)
    test_y_predicted = iris_clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, test_y_predicted)
    accuracies.append(accuracy)

# 視覺化
plt.scatter(range, accuracies)
plt.show()
appr_k = accuracies.index(max(accuracies)) + 1
print(appr_k)
