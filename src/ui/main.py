import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core import load_pdf, split_text, vector_store, graph, config


def process_messages(input_message: str) -> str:
    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config
    ): 
        print(step['messages'][-1].text())
        print("====================================================")
    # print(graph.stream())

st.write("Bienvenue !")


with st.sidebar:
    uploaded_file = st.file_uploader("Importez votre pdf pour commencer à intéragir", accept_multiple_files=False, type='pdf')
if uploaded_file is not None:
    tmp_file = './tmp.pdf'
    with open(tmp_file, 'wb') as file:
        file.write(uploaded_file.getvalue())
        filename = uploaded_file.name
    docs = load_pdf(tmp_file)
    all_splits = split_text(docs)
    vector_store.add_documents(documents=all_splits)


chat = st.container()
if prompt := st.chat_input():
    chat.chat_message('user').write(prompt)
    process_messages(prompt)
    chat.chat_message('assistant').write(f'we are processing')