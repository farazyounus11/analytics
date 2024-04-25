import pandas as pd
import streamlit as st
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(layout="wide")
df = pd.read_csv('creditcard.csv')

st.header("The original credit card fraud data set can be found on Kaggle")
st.header("The original dataset has 31 features. I did simple feature selection and reduced the feature count to 5 ", divider = "red")

st.markdown("## You can see in the pairplots that the yellow and blue labels are linearly separable!")

col1, col2 = st.columns([1, 1])
with col1:
    st.write("## DataFrame")
    st.write(df)

# Display image in the second column
with col2:
    st.write("## Pair Plot")
    st.image("pair.png", caption='Pair Plot of Scammers and Non-Scammers')

X = df.drop(columns=['Class'])
y = df['Class']

def train_and_plot_confusion_matrix(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    st.header("I was able to reduce false negatives (scammers who got away) from 35 to 8!")
    st.markdown("---")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figsize as needed
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

st.sidebar.title("Run or Rerun the Model")
if st.sidebar.button("Run"):
    train_and_plot_confusion_matrix(X, y)
