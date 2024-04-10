from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=43)

clf = DecisionTreeClassifier(criterion = 'entropy',max_depth=5,random_state=23)

model = clf.fit(x_train,y_train)
fig = plt.figure()
_ = tree.plot_tree(clf,feature_names = iris.feature_names,class_names = iris.target_names,filled = True)
fig.show()

y_pred = model.predict(x_test)
print('confusion matrix',confusion_matrix(y_test,y_pred))
print('accuracy',accuracy_score(y_test,y_pred)*100)
print('Report',classification_report(y_test,y_pred))
