import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from src.core import load_pdf, split_doc, vectorize_in_memory, Rag, LlmFactory

st.write("Bienvenue !")

with st.sidebar:
    uploaded_file = st.file_uploader("Importez votre pdf pour commencer à intéragir", accept_multiple_files=False, type='pdf')
    if selected_model := st.selectbox(
        "Quel model souhaitez vous utiliser ?", 
        ('openai', 'mistral'),
        index=None,
        placeholder="Choisissez un model"
        ):
            st.write('model choisi : ', selected_model)
            rag = Rag(LlmFactory.create(selected_model))
            rag._initialize_graph()

if uploaded_file is not None and selected_model is not None:
    tmp_file = './tmp.pdf'
    with open(tmp_file, 'wb') as file:
        file.write(uploaded_file.getvalue())
        file = uploaded_file.name
    docs = load_pdf(tmp_file)
    docs_splitted = split_doc(docs=docs)
    vectorize_in_memory(docs_splitted)
elif selected_model is None:
    st.warning('Selectionnez un model pour traiter le document')


def predict(input_message: str) -> str:
    ai_message = ""
    for idx, step in enumerate(rag.graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=rag.config
    )):
       print(f"=========={idx}============\n")
       print(step["messages"][-1])
       if (step["messages"][-1].type == 'ai'):
           ai_message = step["messages"][-1]
    return ai_message

chat = st.container()
if prompt := st.chat_input():
    chat.chat_message('user').write(prompt)
    answer = predict(prompt)
    chat.chat_message('assistant').write(answer.content)