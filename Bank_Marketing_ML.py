import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define file path
file_path = 'bank-full.csv'

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} was not found. Please check the path and try again.")

# Load dataset
data = pd.read_csv(file_path, sep=';')

# Encode target variable
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Identify categorical and numerical columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Label Encoding for categorical variables before One-Hot Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# One-hot encoding for categorical variables
ohe = OneHotEncoder(drop='first', sparse_output=False)
categorical_encoded = ohe.fit_transform(data[categorical_columns])
categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=ohe.get_feature_names_out(categorical_columns))

# Combine numerical and encoded categorical data
final_data = pd.concat([data[numerical_columns], categorical_encoded_df, data['y']], axis=1)

# Train-test split
X = final_data.drop(columns=['y'])
y = final_data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Data Visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 6))
sns.countplot(x='y', data=data, palette='viridis', ax=axes[0])
axes[0].set_title('Distribution of Target Variable (y)')
sns.heatmap(final_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[1])
axes[1].set_title('Feature Correlation Heatmap')
plt.show()

# Model Training & Evaluation
models = {
    'Naïve Bayes': GaussianNB(),
    'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'\n{name} Performance:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Visualization
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_title(f'Confusion Matrix - {name}')

plt.show()
