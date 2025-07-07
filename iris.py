# Task 1: Classical ML with Scikit-learn
# Dataset: Iris Species Dataset

# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Step 2: Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # Features
y = pd.Series(iris.target, name='species')               # Target labels

# Step 3: Check for missing values (just for demonstration)
# NOTE: The original iris dataset has no missing values, but we simulate them here
X.iloc[0, 0] = np.nan  # Introduce a missing value artificially

# Step 4: Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Replace missing values with column mean
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Step 5: Encode labels (only needed if labels are text — here they are already numeric)
# If labels were strings like 'setosa', we would use:
# encoder = LabelEncoder()
# y_encoded = encoder.fit_transform(y)
y_encoded = y  # Already numeric in this case

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42)

# Step 7: Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = clf.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # macro = average across all classes
recall = recall_score(y_test, y_pred, average='macro')

# Step 10: Print the results
print("Model Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
