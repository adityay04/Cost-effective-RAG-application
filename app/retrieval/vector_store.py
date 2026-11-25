import chromadb
from app.config import settings

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_path
        )

        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]]
    ):
        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            embeddings=embeddings,
            metadatas=[chunk["metadata"] for chunk in chunks]
        )

    def delete_chunks(self, ids: list[str]):
        if ids:
            self.collection.delete(ids=ids)

    def get_chunks_by_source(self, source: str) -> list[str]:
        result = self.collection.get(
            where={"source": source},
            include=[]
        )

        return result["ids"]

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None
    ):
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"]
        )

        return result
