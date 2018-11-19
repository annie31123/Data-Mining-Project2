
from sklearn.cross_validation  import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import pandas as pd

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

accuracies = []

for i in range(1,10):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(i,), random_state=1)
    y_pred = clf.fit(x_train,y_train).predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# 視覺化
plt.scatter(range(1,10), accuracies)
plt.show()
appr_k = accuracies.index(max(accuracies)) + 1
print(appr_k)
