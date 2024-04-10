import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics

digits = load_digits()
print('image shape:',digits.data.shape)
print('label shape:',digits.target.shape)

plt.figure(figsize = (20,40))
for index,(image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('Label:%i\n'%label,fontsize = 20)
plt.show()

x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size = 0.33)
lr = LogisticRegression(random_state=0,max_iter = 10000)

lr.fit(x_train,y_train)

p = lr.predict(x_test)

score = lr.score(x_test,y_test)
print(score)
cm = metrics.confusion_matrix(y_test,p)
print(cm)
