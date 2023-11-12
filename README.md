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
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```




## Output:
## Placement data:
![277169578-0d65163e-89bd-4559-a827-b6c984a8b69c](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/5def67ba-e4ba-4071-b836-3bf5736d0ad0)

## salary data:

![277169604-b50c9d81-07bf-4ea2-bd55-314ae5d7f113](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/c1adb677-cdb7-4fa3-afe1-1752c366491b)

## Checking the null() function:
![277169635-27949900-92a0-468a-bb12-8dcb01a83ec0](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/d27a8cde-9887-4737-b6de-78e6ac58c154)

## Data duplicate:
![277169662-9eef8da7-cdf1-4ce6-86f0-9b6c587c6146](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/032aa3d5-86c5-4c2a-80e2-756eba276764)

## Print data:

 ![277169680-6ae4dffb-18b2-4b82-95dc-6c7f83b8c23d](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/b0f3b1e0-7fdc-4407-8e21-f416bd87aefd)


## Data-status:


![277169700-1810a3d9-e2d4-4dc5-8c46-0215a94b6f8d](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/b4962554-b43f-4d89-82d8-79f3f965a45e)



## Y_prediction array:
![277169740-91beecf1-7253-42fb-a1af-22205d0c25a5](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/7cc1e6f3-f609-4f4d-ba84-92dc15e68bb9)


## Accuracy value:

![277169758-86e41762-900b-463d-814c-90a1d0e84355](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/5c71f820-ffc5-4a66-a5dd-653dcb13e4c5)

## Confusion array:
![277169800-5ec7c8f5-829c-4042-a9e1-a1c9b0595adb](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/e0a26381-7281-4b5e-ab4c-c65064ab4908)

## Classification Report:
![277169830-2125bc90-ce93-4f7b-93b7-c6bf6786f51b](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/ffea6594-f16b-4105-83b4-7dbc2e9a06f7)


## Prediction of LR:

![277169929-024f856e-41c9-44aa-874b-c778bf7f1c28](https://github.com/SubashiniSenniappan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119404951/9a600ede-d0f3-41c8-b0af-8e3aada4dcd1)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
