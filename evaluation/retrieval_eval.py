import json
import math
from pathlib import Path
from statistics import mean

from app.config import settings
from app.retrieval.embeddings import embedding_service
from app.retrieval.vector_store import VectorStore


DATASET_PATH = Path("evaluation/dataset.json")
RESULTS_PATH = Path("evaluation/results/retrieval_results.json")


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return None

    retrieved_relevant = len(set(retrieved[:k]) & relevant)

    return retrieved_relevant / len(relevant)


def hit_rate_at_k(retrieved, relevant, k):
    if not relevant:
        return None

    return float(bool(set(retrieved[:k]) & relevant))


def reciprocal_rank(retrieved, relevant):
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def dcg_at_k(retrieved, relevant, k):
    score = 0.0

    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        relevance = 1 if chunk_id in relevant else 0
        score += relevance / math.log2(rank + 1)

    return score


def ndcg_at_k(retrieved, relevant, k):
    if not relevant:
        return None

    ideal_relevant_count = min(len(relevant), k)

    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    if ideal_dcg == 0:
        return 0.0

    actual_dcg = dcg_at_k(retrieved, relevant, k)

    return actual_dcg / ideal_dcg


def context_precision_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        chunk_id in relevant
        for chunk_id in top_k
    )

    return relevant_count / len(top_k)


def evaluate():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    store = VectorStore()
    results = []

    k = settings.top_k
    collection_count = store.collection.count()

    if collection_count == 0:
        raise ValueError("ChromaDB collection is empty. Ingest documents first.")

    k = min(k, collection_count)

    answerable_questions = 0
    unanswerable_questions = 0

    for item in dataset:
        question_id = item["id"]
        question = item["question"]
        relevant = set(item["relevant_chunk_ids"])

        if not relevant:
            unanswerable_questions += 1
            continue

        answerable_questions += 1

        query_embedding = embedding_service.embed_query(question)

        response = store.search(
            query_embedding=query_embedding,
            top_k=k
        )

        retrieved = response["ids"][0]

        results.append({
            "id": question_id,
            "question": question,
            "relevant_chunk_ids": list(relevant),
            "retrieved_chunk_ids": retrieved,
            "recall_at_k": recall_at_k(
                retrieved,
                relevant,
                k
            ),
            "hit_rate_at_k": hit_rate_at_k(
                retrieved,
                relevant,
                k
            ),
            "mrr": reciprocal_rank(
                retrieved,
                relevant
            ),
            "ndcg_at_k": ndcg_at_k(
                retrieved,
                relevant,
                k
            ),
            "context_precision_at_k": context_precision_at_k(
                retrieved,
                relevant,
                k
            )
        })

    metric_names = [
        "recall_at_k",
        "hit_rate_at_k",
        "mrr",
        "ndcg_at_k",
        "context_precision_at_k"
    ]

    metrics = {}

    for metric_name in metric_names:
        values = [
            result[metric_name]
            for result in results
            if result[metric_name] is not None
        ]

        metrics[metric_name] = round(mean(values), 4) if values else None

    summary = {
        "total_questions": len(dataset),
        "answerable_questions": answerable_questions,
        "unanswerable_questions": unanswerable_questions,
        "evaluated_questions": len(results),
        "k": k,
        "metrics": metrics,
        "results": results
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    print(f"Evaluated questions: {len(results)}")
    print(f"Unanswerable questions: {unanswerable_questions}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()
