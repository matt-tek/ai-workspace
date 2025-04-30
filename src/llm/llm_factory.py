from __future__ import annotations
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

class LlmFactory():
    @staticmethod
    def create(provider: str):
        if provider == 'openai':
            return init_chat_model("gpt-4o-mini", model_provider="openai")
        elif provider == 'mistral':
            return init_chat_model("mistral-large-latest", model_provider="mistralai")
        else:
            raise 'Provider unknonw'