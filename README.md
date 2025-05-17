# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by  : VIKASKUMAR M 
RegisterNumber: 212224220122
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data=pd.read_csv("/content/Employee.csv")
data.head()
```
```
data.tail()
```
```
data.isnull().sum()
```
```
data.info()
```
```
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years","salary"]]
x
```
```
y=data["left"]
y
```
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=100)
dt = DecisionTreeClassifier(criterion='entropy', random_state=100)
dt.fit(x_train, y_train)

```
```
y_pred = dt.predict(x_test)
```
```
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
## Output:
**Head Values**

![image](https://github.com/user-attachments/assets/b75bbbdc-b9fb-4313-9fc6-620617924dee)

**Tail Values**

![image](https://github.com/user-attachments/assets/9f622e74-e73d-43f2-8af8-3550a169c87a)

**Sum-Null Values**

![image](https://github.com/user-attachments/assets/185279fe-efda-406f-9519-82b487038d7a)

**DataInfo**

![image](https://github.com/user-attachments/assets/d544870d-310c-4236-9086-d70438072b43)

**Value Count in Left Column**

![image](https://github.com/user-attachments/assets/dd9cc433-e1e0-4764-abc8-0b9ac7b649c3)

**X-Values**

![image](https://github.com/user-attachments/assets/3b428ac1-a21b-472c-9dc9-3db8b9312036)

**Y-Values**

![image](https://github.com/user-attachments/assets/004bce70-4d8d-4ba3-9980-3ae4a2e9c4d0)

**Training The Model**

![image](https://github.com/user-attachments/assets/868a3e07-4108-4686-815a-da37e00ebef7)

**Accuracy AND Data Prediction**

![image](https://github.com/user-attachments/assets/43d0d9c9-5511-41ab-bd97-f5f944613607)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
