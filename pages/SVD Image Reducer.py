import streamlit as st
import numpy as np
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable the warning
st.markdown("## Upload a Random Photo to Reduce Photo Size Using SVD")
st.markdown("## Apps Such as Instagram/WhatsApp/Zoom reduce Image/Video Size to save user data and data center computation")



uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

A = image
plt.rcParams['figure.figsize'] = [8, 4]

X = np.mean(A, -1); # Convert RGB to grayscale

img = plt.imshow(256-X)
img.set_cmap ('gray')
plt.axis ('off')
plt.show()


U, S, VT = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)

j = 0
for r in (5, 20, 100,200):
    # Construct approximate image
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]

    # Create a new figure
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(256 - Xapprox, cmap='gray')
    ax.axis('off')
    ax.set_title('r = ' + str(r))

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)
