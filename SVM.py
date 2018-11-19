
from sklearn.cross_validation  import train_test_split
from sklearn import metrics, svm
from sklearn.naive_bayes import GaussianNB
import pandas as pd

df2=pd.read_csv("restaurant_30.csv",encoding="big5")
df3=df2[['AC','delicious','decorate','place','service','popular']]

#切分訓練 測試資料
x=df3[['AC','delicious','decorate','place','service']]
y=df3[['popular']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) 
#test_size=0.2  =>  假設他學 8 成, 能準確預估出剩餘2成的幾成?

svc = svm.SVC(kernel='rbf').fit(x_train, y_train).predict(x_test)
accuracy = metrics.accuracy_score(y_test, svc)

print('準確率:',accuracy)