"""
Loan Approval Prediction
Author: Gourav Yadav
"""

import os
import warnings
warnings.filterwarnings("ignore")

import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load
print("Loading dataset...")
path = kagglehub.dataset_download("altruistdelhite04/loan-prediction-problem-dataset")
df = pd.read_csv(os.path.join(path, "train_u6lujuX_CVtuZ9i.csv"))
#
print(f"Shape: {df.shape}")

# Preprocess
df.drop('Loan_ID', axis=1, inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].median())
df = df.dropna()
print(df.isnull().sum())
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100)
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluate
print(f"\nLogistic Regression Accuracy: {accuracy_score(y_test, lr.predict(X_test)):.2%}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.2%}")
print(f"\nRandom Forest Report:\n{classification_report(y_test, rf.predict(X_test), target_names=['Rejected','Approved'])}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, rf.predict(X_test)), annot=True, fmt='d',
            cmap='Blues', xticklabels=['Rejected','Approved'],
            yticklabels=['Rejected','Approved'], ax=axes[0])
axes[0].set_title('Confusion Matrix')

pd.Series(rf.feature_importances_, index=X.columns).sort_values().plot(
    kind='barh', ax=axes[1], color='#3498db')
axes[1].set_title('Feature Importance')

plt.tight_layout()
plt.savefig('loan_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: loan_results.png")