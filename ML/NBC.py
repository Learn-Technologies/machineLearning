##from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dataset = datasets.load_iris()
x = dataset.data
y = dataset.target
##x,y = load_iris(return_x_y = True)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state = 0)
gnb = GaussianNB()
y_pred = gnb.fit(x_train,y_train).predict(x_test)

print("Number of mislabeled points out of a total %d points : %d "%(x_test.shape[0],(y_test!=y_pred).sum()))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred,labels = [0,1,2]))

