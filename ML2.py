import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
dataset=pd.read_excel("C:/Users/asus/Downloads/SBP.xlsx")
print(dataset.head())
#print dataset.columns)

data=dataset.to_numpy()
x=data[:,1:3];
#print ('x size: ', x.shape)
y=data[:,-1];
#print ('y size:

model=LinearRegression().fit(x,y)
r_sq=model.score(x,y)
print(f"\n coefficient of determination: {r_sq}\n") #R2
print(model.intercept_)
print(model.coef_)
y_pred=model.predict(x)
print(f'\n target values:\n {y}')
print(f'\n predicted values: \n {y_pred}')
print(f'\n errors:\n {y-y_pred}')
from sklearn.metrics import mean_squared_error
print('\n mean_squarred_error : ', mean_squared_error(y,y_pred))

""" mean_squarred_error :  102.50347099590346"""