from app.retrieval.vector_store import VectorStore

store = VectorStore()

result = store.collection.get(
    include=["documents", "metadatas"]
)

for chunk_id, document, metadata in zip(
    result["ids"],
    result["documents"],
    result["metadatas"]
):
    print("\nID:", chunk_id)
    print("Source:", metadata["filename"])
    print("Chunk:", metadata["chunk_index"])
    print("Text:", document[:300])