import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("## Upload a Random Photo to Reduce Photo Size Using SVD")
st.markdown("## Apps Such as Instagram/WhatsApp/Zoom reduce Image/Video Size to save user data and data center computation")

uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

    image_resized = image.resize((int(image.width * 0.2), int(image.height * 0.2)))
    A = np.array(image_resized)
    X = np.mean(A, -1)

    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S = np.diag(S)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for i, r in enumerate([5, 20, 100, 200]):
        Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
        ax[i // 2, i % 2].imshow(256 - Xapprox, cmap='gray')
        ax[i // 2, i % 2].axis('off')
        ax[i // 2, i % 2].set_title('r = ' + str(r))

    plt.tight_layout()
    st.pyplot(fig)
