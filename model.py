import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df=pd.read_csv('D:\GIT\Container-Crane-Controller\Container_Crane_Controller_Data_Set.csv')
Speed=np.array(df['Speed'].values).reshape(-1,1)
Angle=np.array(df['Angle'].values).reshape(-1,1)
Attributes=np.concatenate((Speed,Angle),axis=1)
Power=np.array(df['Power'].values)
print(Attributes)

model=LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(Attributes, Power)

Score=model.score(Attributes, Power)
print(Score)

Prediction= model.predict([[100,2]])
print(Prediction)

if (Prediction==2):
    print("High Power")
elif (Prediction==1):
    print("Medium Power")
else:
    print("Low Power")