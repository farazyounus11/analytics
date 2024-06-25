import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

st.set_page_config(
    page_title="Home",layout="wide")

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("### I'm Faraz. I am a data science enthusiast. I first heard about statistics in high school. Ever since, I've always wanted to use models to do cool things like prediction, optimization.")
    st.markdown("### All things are difficult before they are easy ~ Thomas Fuller")

with col2:
    st.image('stats.png', caption='models')



st.write("### The side bar has some of the apps I created!")
st.sidebar.success("Select a page above.")


def get_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_files.append(filename)
    return pdf_files
def main():
    current_directory = os.getcwd()
    pdf_files = get_pdf_files(current_directory)
    selected_file = st.selectbox("Click the Dropdown Menu to Select a Project!", pdf_files)
    if selected_file:
        pdf_path = os.path.join(current_directory, selected_file)
        pdf_viewer(pdf_path)

if __name__ == "__main__":
    main()
