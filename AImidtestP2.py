# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:25:27 2023

@author: Asus
"""
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

import seaborn as sns 

File_path = 'C:\\Users\Asus\Downloads\\'
File_name = 'car_data.csv'

df = pd.read_csv( File_path+File_name)

#Preprocess
df.drop(columns=['User ID'], inplace=True)
#df['Age'].fillna(method='bfill', inplace=True)
#df['AnnualSalary'].fillna(method='bfill', inplace=True)

encoders = []
for i in range(0, len(df.columns)-1):
    enc = LabelEncoder()
    df.iloc[:,i] = enc.fit_transform(df.iloc[:, i])
    encoders.append(enc)
    
x = df.iloc[:, 0:3]
y = df['Purchased']

model = DecisionTreeClassifier(criterion='gini')
model.fit(x,y)


score = model.score(x, y)
print('')
print('Accuracy : ', '{:.2f}'.format(score))

feature = x.columns.tolist()
Data_class = y.tolist()

plt.figure(figsize=(25,20))
_ = plot_tree(model,
              feature_names= feature,
              class_names = Data_class,
              label='all',
              impurity=True,
              precision=3,
              filled=True,
              rounded=True,
              fontsize=16)
 
plt.show()

feature_imp = model.feature_importances_
feature_names = ['Age','AnnualSalary']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x=feature_imp, y=feature_names)
print(feature_imp)