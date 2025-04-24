from dotenv import load_dotenv
from typing import Literal
from typing_extensions import List, TypedDict
from typing_extensions import Annotated

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

class ProcessPdf:
    def __init__(self):
        self.loader = None
        self.doc = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        self.emmbeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.graph_builder = StateGraph(MessagesState)

    def load_pdf(self, file):
        self.loader = PyPDFLoader(file)
        self.docs = self.loader.load()
        self.all_splits = self.text_splitter.split_documents(self.docs)
        self.vector_store.add_documents(documents=self.all_splits)
