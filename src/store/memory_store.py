from langchain_core.vectorstores import InMemoryVectorStore
from src.core import embeddings
from typing_extensions import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

vector_store = InMemoryVectorStore(embeddings)

def vectorize_in_memory(all_split) -> list[str]:
    return vector_store.add_documents(documents=all_split)

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

def retrieve_document_data(query: str, extend_index: int) -> Document:
    return vector_store.similarity_search(query, k=extend_index)