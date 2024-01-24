import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp_id.tokenizer import Tokenizer
from indoNLP.preprocessing import remove_stopwords, replace_slang, replace_word_elongation
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import os



st.title('Sentiment Prediction')

@st.cache_data
def load_tfidf_vectorizer():
    return pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))

@st.cache_data
def load_model():
    return pickle.load(open('model/logreg.pkl', 'rb'))

tfidf_vectorizer = load_tfidf_vectorizer()
model = load_model()

# Load the stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Streamlit input text area for user to input review
user_review = st.text_area("Only type in Bahasa Indonesia\n\nEnter your review here:")

# Button to trigger sentiment prediction
if st.button("Predict"):
    # Check if the user has entered a review
    if user_review:
        # Tokenize and clean the user input
        tokenizer = Tokenizer()
        tokenized_text = tokenizer.tokenize(user_review)

        # Apply preprocessing functions
        processed_text = replace_slang(user_review)
        processed_text = replace_word_elongation(user_review)
        processed_text = remove_stopwords(processed_text)

        # Stemming
        stemmed_text = ' '.join([stemmer.stem(token) for token in tokenized_text])

        # Transform the preprocessed user input using the pre-fitted vectorizer
        user_review_tfidf = tfidf_vectorizer.transform([user_review])

        # Ensure that the number of features is consistent
        if user_review_tfidf.shape[1] == model.coef_.shape[1]:
            # Make prediction using the loaded model
            prediction = model.predict(user_review_tfidf)[0]
            probability = model.predict_proba(user_review_tfidf)[0][prediction]

            # Convert the numeric prediction to text
            sentiment_result = "Positive" if prediction == 1 else "Negative"

            # Display the prediction result
            st.write("Sentiment Prediction:", sentiment_result)
            st.write("Probability:", probability)
        else:
            st.warning("Inconsistent number of features between TF-IDF and the model.")
    else:
        st.warning("Please enter a review before predicting.")

