import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)
# Load models
@st.cache_data
def load_models99():
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    logreg_selected = joblib.load('logreg_selected_model.pkl')
    neural_net = joblib.load('neural_net_model.pkl')
    nb_classifier = joblib.load('naive_bayes_model.pkl')
    selector_weights = joblib.load('selector.pkl')
    return tfidf_vectorizer, logreg_selected, neural_net, nb_classifier, selector_weights

tfidf_vectorizer, logreg_selected, neural_net, nb_classifier, selector_weights = load_models99()


def make_predictions99(tfidf_vectorizer, logreg_selected, neural_net, nb_classifier, selector_weights, texts):
    custom_text_tfidf = tfidf_vectorizer.transform(texts)
    custom_text_selected = selector_weights.transform(custom_text_tfidf)
    logreg_pred = logreg_selected.predict(custom_text_selected)
    neural_net_pred = neural_net.predict(custom_text_selected)
    nb_pred = nb_classifier.predict(custom_text_selected)
    return logreg_pred, neural_net_pred, nb_pred

# Streamlit app
st.title('Text Classification App')
st.markdown("### This Model predicts & labels texts from subjects like Physics, Biology, History etc.")
st.markdown("### Input a text to see what the model predicts")

# Text input
text_input = st.text_area('Enter text here:', '')

# Make predictions when button is clicked
if st.button('Predict'):
    if text_input:
        texts = [text_input]
        logreg_pred, neural_net_pred, nb_pred = make_predictions99(tfidf_vectorizer, logreg_selected, neural_net, nb_classifier, selector_weights, texts)
        st.write("Logistic Regression Prediction:", logreg_pred[0])
        st.write("Neural Network Prediction:", neural_net_pred[0])
        st.write("Naive Bayes Prediction:", nb_pred[0])
    else:
        st.write("Please enter some text before predicting.")





