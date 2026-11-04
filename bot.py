import math
from dataclasses import dataclass, field

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from ingest import build_index, get_collection
from models import configure_models

CONDENSE_PROMPT = """בהינתן היסטוריית השיחה ושאלת ההמשך, נסח מחדש את שאלת ההמשך כשאלה עצמאית ומלאה,
שאפשר להבין בלי ההיסטוריה. אם השאלה כבר עצמאית, החזר אותה כמו שהיא.
החזר רק את השאלה המנוסחת, בלי הסברים.

היסטוריית השיחה:
{history}

שאלת ההמשך: {question}

שאלה עצמאית:"""

ANSWER_PROMPT = """אתה העוזר הווירטואלי של WatchStore, עסק סיטונאי לשעונים.
ענה על השאלה אך ורק לפי קטעי המסמכים שלמטה.
אם התשובה לא מופיעה בקטעים, ענה בדיוק: "{no_answer}"
אל תשתמש בידע כללי, אל תנחש ואל תמציא מספרים, מחירים או מדיניות.
ענה בעברית, בקצרה ובבהירות.

קטעי מסמכים:
{context}

שאלה: {question}

תשובה:"""


@dataclass
class BotAnswer:
    text: str
    standalone_question: str
    sources: list = field(default_factory=list)


def to_cosine(score: float) -> float:
    return 1.0 + math.log(score) if score and score > 0 else -1.0


class WatchStoreBot:
    def __init__(self):
        configure_models()
        collection = get_collection()
        if collection.count() == 0:
            build_index()
            collection = get_collection()
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=Settings.embed_model)
        self.retriever = index.as_retriever(similarity_top_k=config.TOP_K)
        self.llm = Settings.llm

    def condense(self, question: str, history: list) -> str:
        if not config.CONDENSE_QUESTION or not history:
            return question
        history_text = "\n".join(f"{role}: {text}" for role, text in history[-6:])
        prompt = CONDENSE_PROMPT.format(history=history_text, question=question)
        rewritten = self.llm.complete(prompt).text.strip()
        return rewritten or question

    def retrieve(self, question: str) -> list:
        results = []
        for item in self.retriever.retrieve(question):
            cosine = to_cosine(item.score)
            if cosine >= config.SIMILARITY_CUTOFF:
                results.append(NodeWithScore(node=item.node, score=cosine))
        return results

    def answer(self, question: str, history: list | None = None) -> BotAnswer:
        standalone = self.condense(question, history or [])
        nodes = self.retrieve(standalone)

        if not nodes:
            return BotAnswer(text=config.NO_ANSWER_TEXT, standalone_question=standalone)

        context = "\n\n---\n\n".join(
            f"[מקור: {n.node.metadata.get('file_name', 'לא ידוע')}]\n{n.node.get_content()}"
            for n in nodes
        )
        prompt = ANSWER_PROMPT.format(
            no_answer=config.NO_ANSWER_TEXT,
            context=context,
            question=standalone,
        )
        text = self.llm.complete(prompt).text.strip()

        sources = [
            (n.node.metadata.get("file_name", "לא ידוע"), round(n.score, 3))
            for n in nodes
        ]
        return BotAnswer(text=text, standalone_question=standalone, sources=sources)
