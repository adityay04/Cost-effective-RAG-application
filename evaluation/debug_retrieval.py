import json
from pathlib import Path

from app.config import settings
from app.retrieval.embeddings import embedding_service
from app.retrieval.vector_store import VectorStore


dataset_path = Path("evaluation/dataset.json")

dataset = json.loads(
    dataset_path.read_text(encoding="utf-8")
)

store = VectorStore()

all_chunks = store.collection.get(
    include=["documents", "metadatas"]
)

known_ids = set(all_chunks["ids"])

print("Total chunks in database:", len(known_ids))

for item in dataset:
    question = item["question"]
    relevant_ids = set(item["relevant_chunk_ids"])

    query_embedding = embedding_service.embed_query(question)

    result = store.search(
        query_embedding=query_embedding,
        top_k=settings.top_k
    )

    retrieved_ids = result["ids"][0]
    retrieved_documents = result["documents"][0]

    print("\nQuestion:", question)
    print("Ground truth IDs:", relevant_ids)
    print("Ground truth IDs exist:", relevant_ids.issubset(known_ids))
    print("Retrieved IDs:", retrieved_ids)

    for chunk_id, document in zip(
        retrieved_ids,
        retrieved_documents
    ):
        print("Retrieved chunk:", chunk_id)
        print("Text:", document[:300])
