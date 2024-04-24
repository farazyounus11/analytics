import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable the warning
st.markdown("## Upload a Random Photo to Reduce Photo Size Using SVD")
st.markdown("## Apps Such as Instagram/WhatsApp/Zoom reduce Image/Video Size to save user data and data center computation")

uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

    # Resize the image to reduce computational load
    image_resized = image.resize((200, 200))  # Adjust the size as needed
    A = np.array(image_resized)  # Convert PIL image to NumPy array
    X = np.mean(A, -1)  # Convert RGB to grayscale

    # Perform Singular Value Decomposition (SVD)
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S = np.diag(S)

    # Create a single Matplotlib figure
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for i, r in enumerate([5, 20, 100, 200]):
        # Construct approximate image
        Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]

        # Display the image in the corresponding subplot
        ax[i // 2, i % 2].imshow(256 - Xapprox, cmap='gray')
        ax[i // 2, i % 2].axis('off')
        ax[i // 2, i % 2].set_title('r = ' + str(r))

    # Adjust layout and display the Matplotlib figure
    plt.tight_layout()
    st.pyplot(fig)
