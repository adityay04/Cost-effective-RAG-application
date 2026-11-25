import time

from groq import Groq

from app.config import settings
from app.retrieval.embeddings import embedding_service
from app.retrieval.vector_store import VectorStore


client = Groq(api_key=settings.groq_api_key)


def ask_question(
    question: str,
    top_k: int | None = None,
    metadata_filter: dict | None = None
) -> dict:
    start_time = time.perf_counter()

    query_embedding = embedding_service.embed_query(question)

    vector_store = VectorStore()

    result = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k or settings.top_k,
        metadata_filter=metadata_filter
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    relevant_chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        if distance <= 0.65:
            relevant_chunks.append({
                "text": document,
                "metadata": metadata,
                "distance": distance
            })

    if not relevant_chunks:
        return {
            "answer": "I couldn't find relevant information in the provided documents.",
            "citations": [],
            "retrieved_chunks": len(documents),
            "used_chunks": 0,
            "context": "",
            "latency_ms": round(
                (time.perf_counter() - start_time) * 1000,
                2
            ),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    context_parts = []
    citations = []

    for index, chunk in enumerate(relevant_chunks, start=1):
        source_label = f"Source {index}"
        metadata = chunk["metadata"]

        context_parts.append(
            f"[{source_label}]\n"
            f"File: {metadata['filename']}\n"
            f"Chunk: {metadata['chunk_index']}\n"
            f"Content: {chunk['text']}"
        )

        citations.append({
            "source_label": source_label,
            "source": metadata["filename"],
            "chunk_index": metadata["chunk_index"],
            "distance": round(chunk["distance"], 4)
        })

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question using only the supplied context.

Rules:
1. Do not use outside knowledge.
2. If the context does not answer the question, say so.
3. Cite supporting sources using [Source 1], [Source 2], etc.
4. Never invent citations.
5. Keep the answer clear and concise.
"""

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content or ""

    usage = response.usage

    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0

    return {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": len(documents),
        "used_chunks": len(relevant_chunks),
        "context": context,
        "latency_ms": round(
            (time.perf_counter() - start_time) * 1000,
            2
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
