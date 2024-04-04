import pickle
import streamlit as st
import re
import nltk

nltk.download('stopwords')  # Download required stopwords
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def preprocess_text(text):
    
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    stop_words = set(stopwords.words('english'))  # Load stopwords
    tokens = text.split()  # Split into tokens
    filtered_tokens = [w for w in tokens if w not in stop_words]  # Remove stopwords
    ps = PorterStemmer()  # Initialize stemmer
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]  # Stem words
    return " ".join(stemmed_tokens)  # Join tokens back with spaces


def main():
    st.title("Sentiment Analysis App")

    # About section (expander collapsed by default)
    with st.expander("About", expanded=False):
        st.subheader("Model Description")
        st.markdown("This sentiment analysis model is based on a Multinomial Naive Bayes classifier.")
        st.markdown("It was trained on a dataset of labeled tweets to classify the sentiment of a given tweet "
                    "as positive, negative, or neutral.")
        st.markdown("The model preprocesses the input text by removing non-alphanumeric characters, stopwords, "
                    "and stemming words to improve classification accuracy.")

    # Disclaimer section
    st.subheader("Disclaimer")
    st.markdown("This sentiment analysis app is for educational and demonstration purposes only.")
    st.markdown("The predictions made by the model are based on statistical patterns learned from the training data "
                "and may not always accurately reflect the sentiment of all tweets in real-world scenarios.")

    # Input field and analysis
    user_text = st.text_input("Enter a tweet to analyze its sentiment:")

    if user_text:
        try:
            # Load the model and handle vectorizer
            with open("model.pkl", "rb") as f:  # Replace with your actual path
                clf = pickle.load(f)
            if hasattr(clf, 'vectorizer_'):
                vectorizer = clf.vectorizer_
            else:
                # Handle potential lack of saved vectorizer
                raise ValueError("Vectorizer information not found in model.")

            # Preprocess text and predict sentiment
            preprocessed_text = preprocess_text(user_text)
            X = vectorizer.transform([preprocessed_text])
            prediction = clf.predict(X)[0]

            # Display results based on prediction
            if prediction == 0:
                st.success("The tweet is neutral.")
            elif prediction == 1:
                st.success("The tweet is positive!")
            else:
                st.error("The tweet is negative.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please enter a tweet to analyze its sentiment.")

  
if _name_ == "_main_":
    main()