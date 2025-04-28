from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


load_dotenv()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

def load_pdf(filepath: str) -> list[Document]:
    loader = PyPDFLoader(filepath)
    return loader.load()


def split_doc(docs: Document) -> list[Document]:
    return text_splitter.split_documents(docs)

