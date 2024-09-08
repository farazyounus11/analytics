import streamlit as st
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import DecisionBoundaryDisplay

st.set_page_config(layout="wide")
st.markdown('## Classification Machine Learning ')
st.markdown("### Classification modeling can help identify things like credit card fraud, cancer detection, diabetes, and plant/species classification, buyer engagement etc!")

def list_csv_files():
    files = os.listdir()
    return [file for file in files if file.endswith('_Classification.csv')]

# Streamlit app
def main():

    # List all CSV files in the current directory
    csv_files = list_csv_files()
    
    default_file_index = csv_files.index('Credit_Card_Fraud_Classification.csv')
    st.markdown("### Select two unique variables using below to see if they're good at classification problem")

    # Selectbox for file selection in sidebar
    selected_file = st.selectbox('Select a Classification Problem!', csv_files, index= default_file_index)

    if selected_file:
        # Load the selected CSV file
        df = pd.read_csv(selected_file)

        # Define y as df["Y"]
        y = df.pop('Y')
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)


        all_columns = df.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            feature1 = st.selectbox('Select First variable', all_columns)
        
        remaining_columns = [col for col in all_columns if col != feature1]
        
        with col2:
            feature2 = st.selectbox('Select Second variable', remaining_columns)

        X = df[[feature1, feature2]].values

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X,  y,  test_size=0.2)

        # List of classifiers to visualize
        classifiers = [
            ("Logistic Regression", LogisticRegression(C=1.0, penalty='l2', solver='liblinear')),  # L2 regularization (default)
            ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=3)),  # No direct regularization, but the number of neighbors acts as a regularizer
            ("Support Vector Machine", SVC(kernel="linear", C=0.025, probability=True)),  # Regularization via the C parameter
            ("Decision Tree", DecisionTreeClassifier(max_depth=5, min_samples_split=4, min_samples_leaf=2)),  # Regularization via depth and sample splits
            ("Random Forest", RandomForestClassifier(max_depth=5, min_samples_split=4, min_samples_leaf=2)),  # Same as Decision Tree
            ("AdaBoost", AdaBoostClassifier(n_estimators=20, learning_rate=0.5)),  # Learning rate can act as a regularizer
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=3, min_samples_split=4, min_samples_leaf=2)),  # Regularization via learning rate and tree parameters
            ("Neural Network", MLPClassifier(alpha=0.001, max_iter=150))  # L2 regularization with alpha
        ]

        # List of color maps to use
        color_maps = [plt.cm.summer]

        # Create grid for test plots
        fig_test, axes_test = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        axes_test = axes_test.flatten()  # Flatten the 2D array of axes for easy indexing

        # Plot decision boundaries for each classifier and calculate F1 score and accuracy
        for i, (name, clf) in enumerate(classifiers):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Plot decision boundary and scatter plot for testing data
            ax_test = axes_test[i]
            cmap = color_maps[i % len(color_maps)]
            display_test = DecisionBoundaryDisplay.from_estimator(
                clf,
                X_test,
                response_method="auto",  # Changed to 'auto'
                ax=ax_test,
                cmap=cmap,
                xlabel=feature1,
                ylabel=feature2
            )
            ax_test.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, edgecolor='k', s=30, label='Test')
            ax_test.set_title(f"{name} (Test)\nAccuracy: {accuracy:.2f} | F1: {f1:.2f}")
        # Adjust layout and display
        fig_test.tight_layout()

        st.pyplot(fig_test)

if __name__ == '__main__':
    main()
