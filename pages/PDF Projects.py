import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import glob

st.set_page_config(layout="wide")

def get_pdf_files(directory):
    return glob.glob(os.path.join(directory, "*.pdf"))

# Main Streamlit app code
def main():
    st.title("Faraz Younus Data Science Projects in PDF Format")
    current_directory = os.getcwd()
    pdf_files = get_pdf_files(current_directory)

    selected_file = st.selectbox("Click the Dropdown Menu to Select a Project!", pdf_files)

    # Display the selected PDF file
    if selected_file:
        pdf_viewer(selected_file)

if __name__ == "__main__":
    main()
