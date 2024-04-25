import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
df = pd.read_csv('creditcard.csv')

st.header("The original credit card fraud data set can be found on Kaggle")
st.header("The original dataset has 31 features. I did simple feature selection and reduced the feature count 5 ", divider = "red")

st.mardown("## You can in in the pairplots that the yellow and blue labels are linearly seperably!")

## Show data frame 
import streamlit as st

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


st.header("I was able to reduce false negatives(scammers who got away) from 35 to 8!", divider = "red")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

# Display the plot in Streamlit
st.pyplot(fig)


