import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer


st.set_page_config(
    page_title="Home")
st.title("Hi I'm Faraz.")
st.title("I am a Masters Data Science Student!")

st.write("## The side bar has some of the apps I created!")
st.image("stats.png", caption='Different Statistical Models doing Classification!')
st.sidebar.success("Select a page above.")
