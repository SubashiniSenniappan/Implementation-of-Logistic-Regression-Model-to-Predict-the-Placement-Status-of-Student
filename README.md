# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Subashini.S
RegisterNumber:  212222240106
*/



#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset = pd.read_csv('Placement_Data_Full_Class.csv')
dataset

dataset.head(20)

dataset.tail(20)

#droping tha serial no salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('salary',axis=1)

dataset = dataset.drop('gender',axis=1)
dataset = dataset.drop('ssc_b',axis=1)
dataset = dataset.drop('hsc_b',axis=1)
dataset

dataset.shape

dataset.info()

#catgorising col for further labelling
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset.info()

dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

dataset.info()

dataset

#selecting the features and labels
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)
dataset.head()

x_train.shape

x_test.shape

y_train.shape

y_test.shape

from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

clf.predict([[0, 87, 0, 95, 0, 2, 78, 2, 0]])
*/
```
## Output:
## dataset:
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/e5cb4322-7916-47fb-8a8a-169ce450f13c)

## dataset.head():
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/c70b20b2-f32f-4468-bfa9-983b7d36c8e4)

## dataset.tail():
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/09b8cf1a-89f0-423d-9470-d2d7e38832cf)

## dataset after dropping:
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/a6d1419f-c525-456d-8808-2c0e4cbccfa3)
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/bdc3a382-1c02-4464-aaa8-1ea8740a2909)

## datase.shape:
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/df576abc-83a3-43d2-b36e-4f5d5a369bc4)
## dataset.info()
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/557ba222-f8e1-4f99-a1fa-2feca82a6d76)
## dataset.dtypes
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/91ea2cb9-eb11-4d63-b4b4-3608dd7e68d2)
## dataset.info()
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/bb3faf13-c9bf-40ca-a665-a44b8e1bccc1)
## dataset.codes
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/e0729b97-015c-454d-95fd-8b0f84fc6d7a)
## selecting the features and labels
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/11d1c8e6-0c22-4053-8f49-45b4f8eb1dd6)
## dataset.head()
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/7f9aa6c5-d4be-48c6-b0a4-e53d1acb69f2)
## x_train.shape
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/bc68db53-9c95-4358-9806-87a4237ab8ab)
## x_test.shape
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/5747336c-24d7-49ea-8ce1-d82f402f588e)
## y_train.shape
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/844b94f7-1bcf-4098-872e-91cf78308402)
## y_test.shape:
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/54c11a5f-bd32-4781-a87a-2ad803fa8af5)
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/835f914c-e725-42c5-b2cd-c5ff8a64e3f9)
## clf.predict:
![image](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/58b5ca46-6942-44ee-b124-51bf2485ec44)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
