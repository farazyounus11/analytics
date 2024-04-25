import pandas as pd
import streamlit as st
from mlxtend.plotting import plot_decision_regions
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

st.markdown("## You can in in the pairplots that the yellow and blue labels are linearly separable!")

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


st.header("I was able to reduce false negatives(scammers who got away) from 35 to 7!", divider = "red")
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

class_labels = df['Class'].unique()
min_samples = df['Class'].value_counts().min()
equal_samples = []
for label in class_labels:
    class_subset = df[df['Class'] == label].sample(min_samples, replace=False)
    equal_samples.append(class_subset)
dfsam = pd.concat(equal_samples)

st.write(dfsam)
column1 = dfsam['V4'].values.reshape(-1, 1)
column2 = dfsam['V11'].values.reshape(-1, 1)

# Concatenate the two columns into a single matrix
concatenated_matrix = np.concatenate((column1, column2), axis=1)

# Ensure that dfsam.Class is a NumPy array
y_np = dfsam['Class'].values

# Plot decision regions
fig = plt.figure(figsize=(10, 5))
plot_decision_regions(concatenated_matrix, y_np, clf=model)
plt.xlabel('Feature 1')  # Replace with appropriate feature name
plt.ylabel('Feature 2')  # Replace with appropriate feature name
plt.title('Decision Regions for Binary Logistic Regression with Balanced Class Weight')
plt.show()
