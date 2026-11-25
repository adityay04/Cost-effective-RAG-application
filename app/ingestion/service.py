import hashlib

from app.config import settings
from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_text
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.embeddings import embedding_service

def create_chunk_id(source: str, index: int, text: str) -> str:
    value = f"{source}:{index}:{text}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ingest_file(file_path: str) -> dict:
    document = load_document(file_path)

    if not document["text"]:
        return {
            "source": document["source"],
            "chunks_added": 0,
            "message": "Document contains no extractable text"
        }

    text_chunks = chunk_text(
        document["text"],
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap
    )

    chunks = []

    for index, text in enumerate(text_chunks):
        chunk_id = create_chunk_id(
            document["source"],
            index,
            text
        )

        chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": {
                "source": document["source"],
                "filename": document["filename"],
                "file_type": document["file_type"],
                "chunk_index": index
            }
        })

    vector_store = VectorStore()

    embeddings = embedding_service.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    existing_ids = vector_store.get_chunks_by_source(
        document["source"]
    )

    new_ids = {chunk["id"] for chunk in chunks}
    stale_ids = list(set(existing_ids) - new_ids)

    vector_store.upsert_chunks(chunks, embeddings)
    vector_store.delete_chunks(stale_ids)

    return {
        "source": document["source"],
        "chunks_added": len(chunks),
        "stale_chunks_removed": len(stale_ids),
        "message": "Ingestion completed"
    }
