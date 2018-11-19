from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation  import train_test_split
from sklearn import metrics, tree
import pandas as pd

from graphviz import Source
from IPython.display import SVG  


df2=pd.read_csv("restaurant_30.csv",encoding="big5")
df3=df2[['AC','delicious','decorate','place','service','popular']]

#切分訓練 測試資料
x=df3[['AC','delicious','decorate','place','service']]
y=df3[['popular']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) 
#test_size=0.2  =>  假設他學 8 成, 能準確預估出剩餘2成的幾成?

# 建立分類器
Dtree=DecisionTreeClassifier() 
#criterion : string, optional (default=”gini”) / entropy

tree_clf=Dtree.fit(x_train,y_train)

# 預測
y_test_predicted = tree_clf.predict(x_test)
print('預測結果：')
print(y_test_predicted)

# 標準答案
print('標準答案：')
print(y_test)

accuracy = metrics.accuracy_score(y_test, y_test_predicted)
print('準確率:',accuracy)

graph = Source( tree.export_graphviz(tree_clf, out_file=None,feature_names=['AC','delicious','decorate','place','service'], class_names='popular'))
graph.format = 'png'
graph.render('dtree_render',view=True)