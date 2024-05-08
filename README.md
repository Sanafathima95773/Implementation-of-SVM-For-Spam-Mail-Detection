# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1 import neccessary libraries required.
2 Load the dataset using pd.read_csv.
3 Use CountVectorizer to convert text data into a matrix of token counts.
4 Create an SVM model with a linear kernel.
5 print the accuracy and classification report

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sana Fathima H
RegisterNumber:  212223240145
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as t
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
df=pd.read_csv("/content/spam.csv",encoding='ISO-8859-1')
df.head()

vectorizer=CountVectorizer()
x=vectorizer.fit_transform(df['v2'])
y=df['v1']
x_train,x_test,y_train,y_test=t(x,y,test_size=0.25,random_state=42)
model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)
predictions=model.predict(x_test)
print("accuracy:",accuracy_score(y_test,predictions))
print("Classification report:")
print(classification_report(y_test,predictions))
```

## Output:
![image](https://github.com/Sanafathima95773/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147084627/15ae8bb4-b248-4f44-8ed8-ca34828516bf)
![image](https://github.com/Sanafathima95773/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147084627/b993cb20-55d6-4228-963e-00ad1acb1ad4)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
