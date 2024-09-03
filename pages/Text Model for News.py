import streamlit as st
import joblib
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)


#label_to_category = {1: "World",2: "Sports",3: "Business",4: "Sci/Tech"}

#selected_category_label = st.radio("Select a category", list(label_to_category.values()))
#selected_true_label = [key for key, value in label_to_category.items() if value == selected_category_label][0]
#st.write("Selected true label:", selected_true_label)


porter_stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\b\d+\b', '', text)  
    text = re.sub(r"[^\w\s']", ' ', text.lower()) 
    tokens = [porter_stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def preprocess_texts(texts):
    preprocessed_texts = [preprocess_text(text) for text in texts]
    return preprocessed_texts

def transform_texts(preprocessed_texts, tfidf_vectorizer):
    custom_text_tfidf = tfidf_vectorizer.transform(preprocessed_texts)
    return custom_text_tfidf

def _predict(_nb_classifier, logreg_classifier, custom_text_tfidf):
    try:
        nb_pred = _nb_classifier.predict(custom_text_tfidf)
        logreg_pred = logreg_classifier.predict(custom_text_tfidf)
        return nb_pred, logreg_pred
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None, None

@st.cache_data
def load_models():
    try:
        nb_classifier = joblib.load('nb_classifier1.pkl')
        logreg_classifier = joblib.load('logreg_classifier1.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer1.pkl')
        return nb_classifier, logreg_classifier, tfidf_vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None


st.title('News Classification App')
st.markdown("### This simple model can help companies identify Business, Sports, World, Tech/Science articles")
st.markdown("### Paste a random news article to see what model classifies the text")

# Text input
text_input = st.text_area('Enter text here:', '')

# Make predictions when button is clicked
label_to_category = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

# Make predictions when button is clicked
if st.button('Predict'):
    if text_input:
        texts = [text_input]
        nb_classifier, logreg_classifier, tfidf_vectorizer = load_models()
        preprocessed_texts = preprocess_texts(texts)
        custom_text_tfidf = transform_texts(preprocessed_texts, tfidf_vectorizer)
        nb_pred, logreg_pred = _predict(nb_classifier, logreg_classifier, custom_text_tfidf)
        if nb_pred is not None and logreg_pred is not None:
            # Map numeric predictions to category names
            nb_category = label_to_category.get(nb_pred[0], "Unknown")
            logreg_category = label_to_category.get(logreg_pred[0], "Unknown")
            st.write("Naive Bayes Prediction:", nb_category)
            st.write("Logistic Regression Prediction:", logreg_category)
    else:
        st.write("Please enter some text before predicting.")
