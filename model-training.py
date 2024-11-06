# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.utils import resample
import numpy as np
import pickle
from scipy.spatial.distance import jensenshannon

import kagglehub

# Download latest version of spam_Emails_data
path = kagglehub.dataset_download("meruvulikith/190k-spam-ham-email-dataset-for-classification")
print("Path to dataset files:", path)

# Load the CSV files
print("Loading CSV files...")
file_paths = {
    "combined_data": "data/combined_data.csv",
    "lingSpam": "data/lingSpam.csv",
    "completeSpamAssassin": "data/completeSpamAssassin.csv",
    "spam_ham_dataset": "data/spam_ham_dataset.csv",
    "spam_Emails_data": kagglehub.dataset_download("meruvulikith/190k-spam-ham-email-dataset-for-classification")
}

# Read each CSV file
dataframes = {}
for name, path in file_paths.items():
    print(f"Reading {name} from {path}...")
    dataframes[name] = pd.read_csv(path)
    print(f"Loaded {name}, shape: {dataframes[name].shape}")

# Standardize column names and add source labels
for name, df in dataframes.items():
    print(f"Processing {name} dataframe...")
    if 'Body' in df.columns:
        df.rename(columns={'Body': 'text', 'Label': 'label'}, inplace=True)
    elif 'CONTENT' in df.columns and 'CLASS' in df.columns:
        df.rename(columns={'CONTENT': 'text', 'CLASS': 'label'}, inplace=True)
    df = df[['text', 'label']]
    df.loc[:, 'source'] = name  # Use .loc to avoid SettingWithCopyWarning
    dataframes[name] = df
    print(f"Processed {name}, columns: {df.columns.tolist()}")

# Concatenate all dataframes
print("Concatenating all dataframes...")
data = pd.concat(dataframes.values(), ignore_index=True)
print(f"Concatenated data shape: {data.shape}")

# Encode labels as 0 (ham) and 1 (spam)
print("Encoding labels...")
data['label'] = data['label'].apply(lambda x: 1 if x in [1, 'spam', 'spam '] else 0)
print("Label encoding completed. Label distribution:")
print(data['label'].value_counts())

# Drop rows with missing text data
print("Dropping rows with missing text data...")
data.dropna(subset=['text'], inplace=True)
print(f"Data shape after dropping missing text: {data.shape}")

# Balance the dataset to have equal ham and spam samples
print("Balancing the dataset...")
spam_data = data[data['label'] == 1]
ham_data = data[data['label'] == 0]
print(f"Spam data shape: {spam_data.shape}, Ham data shape: {ham_data.shape}")

# Resample the minority class to match the majority class size
if len(spam_data) > len(ham_data):
    print("Resampling spam data...")
    spam_data = resample(spam_data, replace=False, n_samples=len(ham_data), random_state=42)
else:
    print("Resampling ham data...")
    ham_data = resample(ham_data, replace=False, n_samples=len(spam_data), random_state=42)

balanced_data = pd.concat([spam_data, ham_data])
print(f"Balanced data shape: {balanced_data.shape}")

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(balanced_data['text'], balanced_data['label'], test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Convert text data to TF-IDF features
print("Converting text data to TF-IDF features...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF transformation completed.")

# Define logistic regression model and hyperparameter grid
print("Defining logistic regression model and hyperparameter grid...")
model = LogisticRegression(solver='liblinear', random_state=42)
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}
print(f"Hyperparameter grid: {param_grid}")

# Perform hyperparameter optimization with GridSearchCV
print("Performing hyperparameter optimization with GridSearchCV...")
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)
print("Grid search completed.")

# Best model from grid search
print("Best parameters found:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Cross-validation with the best model
print("Performing cross-validation with the best model...")
cross_val_scores = cross_val_score(best_model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f'Cross-validation scores: {cross_val_scores}')
mean_cross_val_accuracy = cross_val_scores.mean()
print(f'Mean cross-validation accuracy: {mean_cross_val_accuracy:.2f}')

# Check if accuracy drops below 80%
if mean_cross_val_accuracy < 0.80:
    print("WARNING: Cross-validation accuracy has dropped below 80%!")

# Predict on the test set
print("Predicting on the test set...")
y_pred = best_model.predict(X_test_tfidf)
y_pred_proba = best_model.predict_proba(X_test_tfidf)[:, 1]
print("Prediction completed.")

# Calculate accuracy and log loss
print("Calculating accuracy and log loss...")
accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_pred_proba)
print(f'Accuracy: {accuracy:.2f}')
if accuracy < 0.80:
    print("WARNING: Test set accuracy has dropped below 80%!")
print(f'Log Loss: {loss:.2f}')

# Confusion Matrix
print("Generating confusion matrix...")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("Generating classification report...")
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# ROC AUC Score
print("Calculating ROC AUC Score...")
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC Score: {roc_auc:.2f}')

# Data Drift Monitoring using Jensen-Shannon Divergence
print("Monitoring data drift using Jensen-Shannon Divergence...")
train_features_mean = np.mean(X_train_tfidf.toarray(), axis=0)
test_features_mean = np.mean(X_test_tfidf.toarray(), axis=0)

# Calculate the Jensen-Shannon Divergence between training and test distributions
js_divergence = jensenshannon(train_features_mean, test_features_mean)
print(f'Jensen-Shannon Divergence between training and test feature distributions: {js_divergence:.4f}')

# Alert if JS divergence is above a reasonable threshold indicating potential data drift
js_threshold = 0.2
if js_divergence > js_threshold:
    print("WARNING: Significant data drift detected! Jensen-Shannon Divergence is above the threshold.")

# Save the model and vectorizer for later use
print("Saving the trained model and vectorizer...")
with open('model/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Model and vectorizer saved successfully.")
