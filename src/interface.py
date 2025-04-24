import streamlit as st
import pandas as pd
from services.rag import ProcessPdf

st.write("Bienvenue !")
processor = ProcessPdf

with st.sidebar:
    uploaded_file = st.file_uploader("Importez votre pdf pour commencer à intéragir", accept_multiple_files=False, type='pdf')
if uploaded_file is not None:
    tmp_file = './tmp.pdf'
    with open(tmp_file, 'wb') as file:
        file.write(uploaded_file.getvalue())
        filename = uploaded_file.name
        processor.load_pdf(filename)


chat = st.container()
if prompt := st.chat_input():
    chat.chat_message('user').write(prompt)
    chat.chat_message('assistant').write(f'we are processing')