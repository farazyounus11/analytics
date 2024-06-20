import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

!pip install --upgrade numpy pandas
st.set_page_config(
    page_title="Home",layout="wide")

st.title("Hi I'm Faraz.")
st.title("I am a Masters Data Science Student!")
st.write("## The side bar has some of the apps I created!")
st.write("## Check out my projects below")
st.image("stats.png", caption='Different Statistical Models doing Classification!')
st.sidebar.success("Select a page above.")

def get_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_files.append(filename)
    return pdf_files

# Main Streamlit app code
def main():
    st.title("Faraz Younus Data Science Projects in PDF Format")
    current_directory = os.getcwd()
    pdf_files = get_pdf_files(current_directory)

    selected_file = st.selectbox("Click the Dropdown Menu to Select a Project!", pdf_files)

    # Display the selected PDF file
    if selected_file:
        pdf_path = os.path.join(current_directory, selected_file)
        pdf_viewer(pdf_path)

if __name__ == "__main__":
    main()
