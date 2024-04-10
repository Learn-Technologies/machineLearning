from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score

iris = datasets.load_iris()
x = iris.data
y = iris.target

err = []
for i in range(1,11):
        est = KMeans(n_clusters = i,n_init =25,init = 'k-means++',random_state = 0)
        pred_y = est.fit_predict(x)
        err.append(est.inertia_)
plt.plot(range(1,11),err)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Error')

plt.show()

fest = KMeans(n_clusters=3,n_init=25,init='k-means++',max_iter=1500,random_state=0,tol=1e-06)
pred_y = fest.fit_predict(x)

plt.scatter(x[pred_y==0,0],x[pred_y==0,1],s=100,c='red',label='sentosa')
plt.scatter(x[pred_y==1,0],x[pred_y==1,1],s=100,c='blue',label='versicolor')
plt.scatter(x[pred_y==2,0],x[pred_y==2,1],s=100,c='green',label='verginical')

plt.scatter(fest.cluster_centers_[:,0],fest.cluster_centers_[:,1],s=100,c='yellow',label = 'Centroid')

plt.legend()
plt.show()

print('Score = ',homogeneity_score(y,pred_y))

