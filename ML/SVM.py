import sklearn as sk
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
dataset = datasets.load_iris()

x = dataset.data
y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
svm_mode = svm.SVC(kernel = 'linear')
svm_mode.fit(x_train,y_train)

dataClass = svm_mode.predict(x_test)
print("The iris type is:")
print(dataset.target_names[dataClass])
print(accuracy_score(y_test,dataClass))
print(confusion_matrix(y_test,dataClass,labels = [0,1,2]))



