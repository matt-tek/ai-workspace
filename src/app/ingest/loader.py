from typing_extensions import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdf(filepath: str) -> List[Document]:
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    return documents

def split_text(documents: List[Document], chunk_size: int = 1000, overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=overlap,
            add_start_index=True
        )
    all_splits = text_splitter.split_documents(documents=documents)
    return all_splits

