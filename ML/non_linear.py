import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

df1 = pd.read_excel('E:/msc_14/Dataset_ML/Fuel_Consumption_Ratings.xlsx')
df=df1.dropna()
print(df)
df_x2 = df[['Unnamed: 11']]
df_y = df[['CO2 Emissions']]

plt.scatter(df_x2,df_y,color='blue')
plt.xlabel('fuel_consumption')
plt.ylabel('CO2_Emission')
plt.show()

poly_features = PolynomialFeatures(degree=3)
df_x = poly_features.fit_transform(df_x2)

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,random_state=42,test_size = 0.33)
reg1 = linear_model.LinearRegression()
reg1.fit(x_train,y_train)
print('coefficient of data:',reg1.coef_)
print('intercept of data:',reg1.intercept_)

test_y_h = reg1.predict(x_test)
np.mean(np.absolute(test_y_h - y_test))
msq1 = np.mean((test_y_h - y_test)**2)
print("errror = ",msq1)

r2score = r2_score(test_y_h,y_test)
print("Score is",r2score)
