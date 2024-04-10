import sklearn as sk
from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

dataset = datasets.load_iris()
print(dataset)

x = dataset.data
y = dataset.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state = 42)
model = sk.neighbors.KNeighborsClassifier(n_neighbors=2,weights="distance",metric="manhattan")
model.fit(x_train,y_train)
dataclass = model.predict(x_test)
print("iris type is: ")
print(dataset.target_names[dataclass])
print(accuracy_score(y_test,dataclass))
print(confusion_matrix(y_test,dataclass,labels=[0,1,2]))
