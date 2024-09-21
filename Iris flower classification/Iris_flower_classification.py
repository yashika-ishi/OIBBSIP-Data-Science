import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('C:\\Users\\YASHIKA NEGI\\Documents\\Yashika\\VS code\\vs code python\\OIBSIP-Data Science\\Task 1\\Iris.csv')

# Drop id column
df.drop('Id', axis=1, inplace=True)

# Correlation Matrix
numerical_columns = df.select_dtypes(include=['number'])
correlation_matrix = numerical_columns.corr()
plt.figure(figsize=(10,5))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn')
plt.title("Correlation of Iris dataset")
plt.show()

# Basic Information
df.info()
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])
df.describe().T
df['Species'].value_counts()
df.nunique()

# Check Missing Values
missing = df.isnull().sum().sort_values(ascending=False)
print("Missing values:\n", missing)

# Drop Duplicate Values
print("Number of duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Visualize Outliers
plt.figure(figsize=(8, 6))
sns.boxplot(data=df)
plt.title("Data with Outliers")
plt.show()

# KDE Plots
sns.kdeplot(df['SepalWidthCm'], fill=True, label="SepalWidthCm")
sns.kdeplot(df['SepalLengthCm'], fill=True, label="SepalLengthCm")
plt.legend()
plt.show()

sns.kdeplot(df['PetalWidthCm'], fill=True, label="PetalWidthCm")
sns.kdeplot(df['PetalLengthCm'], fill=True, label="PetalLengthCm")
plt.legend()
plt.show()

# Model Training
X = df.drop(['Species'], axis=1)
y = df['Species']

# Encode Species if not numerical
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Logistic Regression model
lg = LogisticRegression(max_iter=200)
clf = lg.fit(X_train, y_train)

# Predict the model
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
import pickle
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(clf, file)
