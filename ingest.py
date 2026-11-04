import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from models import configure_models


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if reset:
        existing = [c.name for c in client.list_collections()]
        if config.COLLECTION_NAME in existing:
            client.delete_collection(config.COLLECTION_NAME)
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index() -> int:
    configure_models()

    documents = SimpleDirectoryReader(
        input_dir=str(config.DATA_DIR),
        recursive=True,
        required_exts=[".md", ".txt", ".pdf", ".docx"],
    ).load_data()

    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    collection = get_collection(reset=True)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=Settings.embed_model)

    print(f"נטענו {len(documents)} מסמכים, נוצרו {len(nodes)} chunks ונשמרו ב-ChromaDB.")
    return len(nodes)


if __name__ == "__main__":
    build_index()
