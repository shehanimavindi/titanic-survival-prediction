import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt 


data = pd.read_csv(r'D:\DATA SCIENCE(Shehani)\Semester 1\Mathematics for Computing\Vs codes\codes vs\script01\Project 1\titanic.csv')

print(data.head())
print(data.info())

data.dropna()

from sklearn.model_selection import train_test_split 

X = data.drop(['Survived'], axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

train_data = X_train.join(y_train)
train_data.hist(figsize = (10,6))
plt.suptitle("Titanic Dataset - Feature Distributions", fontsize =16)
plt.xlabel("value")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

import seaborn as sns
sns.heatmap(data.corr(), annot=True)
plt.show()

from sklearn.ensemble import RandomForestClassfier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

model = RandomForestClassfier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

