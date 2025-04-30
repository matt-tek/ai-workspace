from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

class Embedder():
    @staticmethod
    def emmbed(provider: str):
        if provider == 'openai':
            return OpenAIEmbeddings(model="text-embedding-3-large")
        elif provider == 'mistral':
            return MistralAIEmbeddings(model="mistral-embed")
        else:
            raise 'Embedder unknown'