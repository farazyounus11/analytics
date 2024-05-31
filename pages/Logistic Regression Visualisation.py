import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="white")

st.title("Logistic Regression Visualisation")

# Sidebar elements
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df)

    columns = df.columns.tolist()
    
    # Select target column
    target_column = st.sidebar.selectbox("Select the target column", columns)

    # Select independent variables
    features = st.sidebar.multiselect("Select the independent variables", columns, default=columns[:-1])

    if target_column and features:
        X = df[features]
        y = df[target_column]

        # Convert target column to numerical categorical values
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        if X.empty or y.size == 0:
            st.error("Data contains only NaN or non-numeric values after cleaning. Please check your data and try again.")
        elif len(X) < 2:
            st.error("Not enough data to perform train-test split. Please check your data and try again.")
        else:
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Choose regularization parameter
            regularization = st.sidebar.slider("Regularization parameter (C)", 0.01, 10.0, 1.0)

            # Train classifier
            classifier = LogisticRegression(C=regularization)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)

            st.write(f"Accuracy: {accuracy}")

            # Plot probability contours (for 2D features)
            if len(features) == 2:
                x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
                y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
                xx, yy = np.mgrid[x_min:x_max:.01, y_min:y_max:.01]
                grid = np.c_[xx.ravel(), yy.ravel()]
                probs = classifier.predict_proba(grid)[:, 1].reshape(xx.shape)

                f, ax = plt.subplots(figsize=(8, 6))
                contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
                ax_c = f.colorbar(contour)
                ax_c.set_label("$P(y = 1)$")
                ax_c.set_ticks([0, .25, .5, .75, 1])

                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white", linewidth=1)

                ax.set(aspect="equal", xlim=(x_min, x_max), ylim=(y_min, y_max), xlabel=features[0], ylabel=features[1])
                st.pyplot(f)

            else:
                st.write("Probability contour plot is only available for 2D feature space.")
