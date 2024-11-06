
# main.py - Main Script to Call Training and Evaluation
import pandas as pd
from model import train_model
from evaluation import evaluate_model

# Load the CSV files
file_paths = {
    "combined_data": "/kaggle/input/email-spam-classification-dataset/combined_data.csv",
    "lingSpam": "/kaggle/input/email-spam-dataset/lingSpam.csv",
    "completeSpamAssassin": "/kaggle/input/email-spam-dataset/completeSpamAssassin.csv",
    "spam_ham_dataset": "/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv",
    "spam_Emails_data": "/kaggle/input/190k-spam-ham-email-dataset-for-classification/spam_Emails_data.csv"
}

# Read and preprocess the data
dataframes = {}
for name, path in file_paths.items():
    df = pd.read_csv(path)
    if 'Body' in df.columns:
        df.rename(columns={'Body': 'text', 'Label': 'label'}, inplace=True)
    elif 'CONTENT' in df.columns and 'CLASS' in df.columns:
        df.rename(columns={'CONTENT': 'text', 'CLASS': 'label'}, inplace=True)
    df = df[['text', 'label']]
    df.loc[:, 'source'] = name
    dataframes[name] = df

# Concatenate all dataframes
data = pd.concat(dataframes.values(), ignore_index=True)

# Encode labels as 0 (ham) and 1 (spam)
data['label'] = data['label'].apply(lambda x: 1 if x in [1, 'spam', 'spam '] else 0)

# Drop rows with missing text data
data.dropna(subset=['text'], inplace=True)

# Balance the dataset to have equal ham and spam samples
spam_data = data[data['label'] == 1]
ham_data = data[data['label'] == 0]
if len(spam_data) > len(ham_data):
    spam_data = spam_data.sample(len(ham_data), random_state=42)
else:
    ham_data = ham_data.sample(len(spam_data), random_state=42)

balanced_data = pd.concat([spam_data, ham_data])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_model(balanced_data)

# Evaluate the model
evaluate_model(X_train, X_test, y_train, y_test)
