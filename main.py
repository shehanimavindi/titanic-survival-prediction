import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder # Convert categorical columns into numbers
from sklearn.model_selection import train_test_split #s Split features and target


# Load dataset
data = pd.read_csv(r'D:\DATA SCIENCE(Shehani)\Semester 1\Mathematics for Computing\Vs codes\codes vs\script01\Project 1\titanic.csv')

# Show basic info
print(data.head())
print(data.info())

# Remove missing values
data = data.dropna()

# Covert into numbers
encoder = LabelEncoder()

data['Sex'] = encoder.fit_transform(data['Sex'])
data['Embarked'] = encoder.fit_transform(data['Embarked'])

# Split features and target
X = data.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Histograms
train_data = X_train.join(y_train)

train_data.hist(figsize=(10, 6))
plt.suptitle("Titanic Dataset - Feature Distributions", fontsize=15)
plt.tight_layout()
plt.show()

# Used Correlation heatmap which helps to show the relationship between columns, this helps to detect the important columns and features
plt.figure(figsize=(10, 6))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Used Random Forest Model as it combines many predictions and chooses the majority answer, as this reduces overfitting and work well for large datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Final results
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
