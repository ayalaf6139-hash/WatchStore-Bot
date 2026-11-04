from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

import config


def configure_models() -> None:
    if not config.GOOGLE_API_KEY:
        raise RuntimeError("חסר GOOGLE_API_KEY. העתיקי את .env.example ל-.env והכניסי את המפתח.")

    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=config.EMBED_MODEL,
        api_key=config.GOOGLE_API_KEY,
    )
    Settings.llm = GoogleGenAI(
        model=config.LLM_MODEL,
        api_key=config.GOOGLE_API_KEY,
        temperature=0.1,
    )
