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
st.markdown('## Classification Datasets')
st.markdown("### Classification modeling can help identify things like credit card fraud, cancer detection, diabetes, and plant/species classification, buyer engagement etc!")

def list_csv_files():
    files = os.listdir()
    return [file for file in files if file.endswith('_Classification.csv')]

# Streamlit app
def main():

    # List all CSV files in the current directory
    csv_files = list_csv_files()
    
    default_file_index = list_csv_files.index(1)
    st.markdown("### Select two unique variables using sidebar to see if they're good at classifying a selected dataset")

    # Selectbox for file selection in sidebar
    selected_file = st.selectbox('Select a Classification Dataset!', csv_files, index= default_file_index)

    if selected_file:
        # Load the selected CSV file
        df = pd.read_csv(selected_file)

        # Define y as df["Y"]
        y = df.pop('Y')
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)


        all_columns = df.columns.tolist()

        feature1 = st.selectbox('Select first feature', all_columns)
        remaining_columns = [col for col in all_columns if col != feature1]
        feature2 = st.selectbox('Select second feature', remaining_columns)

        # Extract X (features) based on user selection
        X = df[[feature1, feature2]].values

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X,  y,  test_size=0.2)

        # List of classifiers to visualize
        classifiers = [
            ("Logistic Regression", LogisticRegression()),
            ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=3)),
            ("Support Vector Machine", SVC(kernel="linear", C=0.025, probability=True)),
            ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
            ("Random Forest", RandomForestClassifier(max_depth=5)),
            ("AdaBoost", AdaBoostClassifier(n_estimators=20)),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=20)),
            ("Neural Network", MLPClassifier(alpha=1, max_iter=300))
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
