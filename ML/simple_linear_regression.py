import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
##import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("F:/Sem-2-Materials/AI-ML/vikas-ml/co2_emission_simple.csv")
fe_set = df[['CO2_Emission','City']]
fe_set.head()

plt.scatter(fe_set.City,fe_set.CO2_Emission,color = 'blue')
plt.xlabel('Fuel consumption')
plt.ylabel('Co2 Emission')
plt.show()

df_x = df[['City']]
df_y = df[['CO2_Emission']]

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,random_state=42,test_size = 0.33)
print(x_train.head())
reg1 = linear_model.LinearRegression()
reg1.fit(x_train,y_train)
print('coefficient of data:',reg1.coef_)
print('intercept of data:',reg1.intercept_)
print('singular of data:',reg1.singular_)
print('rank of data:',reg1.rank_)

test_y_h = reg1.predict(x_test)
np.mean(np.absolute(test_y_h - y_test))
msq1 = np.mean((test_y_h - y_test)**2)
print("errror = ",msq1)

r2score = r2_score(test_y_h,y_test)
print("Score is",r2score)
