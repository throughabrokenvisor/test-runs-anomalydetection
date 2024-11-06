# main.py
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load the model and vectorizer
with open('model/logistic_regression_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit input for user prediction
st.title("Spam vs Ham Email Classification")
st.write("## Predict if your email is spam or ham")
user_input = st.text_area("Enter the email content:")
if st.button("Predict"):
    if user_input:
        print("Transforming user input using TF-IDF...")
        user_input_tfidf = vectorizer.transform([user_input])
        print("Making prediction for user input...")
        prediction = best_model.predict(user_input_tfidf)[0]
        prediction_proba = best_model.predict_proba(user_input_tfidf)[0, 1]
        if prediction == 1:
            print(f"The email is predicted to be SPAM with a probability of {prediction_proba:.2f}.")
            st.write(f"The email is predicted to be **SPAM** with a probability of {prediction_proba:.2f}.")
        else:
            print(f"The email is predicted to be HAM with a probability of {1 - prediction_proba:.2f}.")
            st.write(f"The email is predicted to be **HAM** with a probability of {1 - prediction_proba:.2f}.")
    else:
        print("No input provided for prediction.")
        st.write("Please enter email content to predict.")


