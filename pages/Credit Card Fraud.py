import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('creditcard.csv')

st.header("The original credit card fraud data set can be found on Kaggle")
st.header("The original dataset has 31 features. I did simple feature selection and reduced the count to 5 feature", divider = "red")




## Sampling from df 
sample_class_1 = df[df['Class'] == 1].sample(492, random_state=42)
sample_class_0 = df[df['Class'] == 0].sample(500, random_state=42)
sampled_df = pd.concat([sample_class_1, sample_class_0], axis=0)
sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

## Show data frame 
import streamlit as st

# Define layout using columns
col1, col2 = st.columns([1, 1])

# Display DataFrame in the first column
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
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
st.pyplot(fig)



