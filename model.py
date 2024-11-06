
# model.py - Training Model
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

def train_model(data):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

    # Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Define logistic regression model and hyperparameter grid
    model = LogisticRegression(solver='liblinear', random_state=42)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }

    # Perform hyperparameter optimization with GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_tfidf, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Save the best model and vectorizer
    with open('best_model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)
    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    return X_train_tfidf, X_test_tfidf, y_train, y_test
