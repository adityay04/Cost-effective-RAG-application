import csv
import json
import math
import time
from pathlib import Path
from statistics import mean

from app.config import settings
from app.retrieval.embeddings import embedding_service
from app.retrieval.vector_store import VectorStore


DATASET_PATH = Path("evaluation/dataset.json")
RESULTS_PATH = Path("evaluation/results/cost_latency_results.json")
CSV_PATH = Path("evaluation/results/cost_comparison.csv")
ASSUMPTIONS_PATH = Path("evaluation/cost_assumptions.json")

VECTOR_COUNTS = [100_000, 1_000_000, 10_000_000]

EMBEDDING_DIMENSIONS = 384
FLOAT_BYTES = 4
METADATA_BYTES_PER_VECTOR = 500
INDEX_OVERHEAD_MULTIPLIER = 1.5

GROQ_INPUT_PRICE_PER_MILLION = 0.15
GROQ_OUTPUT_PRICE_PER_MILLION = 0.60

MONTHLY_QUERIES = 10_000
AVERAGE_INPUT_TOKENS_PER_QUERY = 1_000
AVERAGE_OUTPUT_TOKENS_PER_QUERY = 200

BENCHMARK_REPEATS = 5
WARMUP_RUNS = 2


def percentile(values, percentile_value):
    if not values:
        return None

    sorted_values = sorted(values)

    index = math.ceil(
        percentile_value / 100 * len(sorted_values)
    ) - 1

    index = max(0, min(index, len(sorted_values) - 1))

    return sorted_values[index]


def load_dataset():
    return json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )


def benchmark_retrieval():
    dataset = load_dataset()
    store = VectorStore()

    if store.collection.count() == 0:
        raise ValueError("ChromaDB collection is empty.")

    questions = [
        item["question"]
        for item in dataset
        if item.get("relevant_chunk_ids")
    ]

    if not questions:
        raise ValueError("No answerable evaluation questions found.")

    query_embeddings = [
        embedding_service.embed_query(question)
        for question in questions
    ]

    for _ in range(WARMUP_RUNS):
        for embedding in query_embeddings:
            store.search(
                query_embedding=embedding,
                top_k=settings.top_k
            )

    latencies = []
    retrieved_counts = []

    for _ in range(BENCHMARK_REPEATS):
        for embedding in query_embeddings:
            start = time.perf_counter()

            result = store.search(
                query_embedding=embedding,
                top_k=settings.top_k
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            latencies.append(elapsed_ms)

            retrieved_counts.append(
                len(result["ids"][0])
            )

    return {
        "benchmark_questions": len(questions),
        "benchmark_repeats": BENCHMARK_REPEATS,
        "total_queries": len(latencies),
        "p50_retrieval_latency_ms": round(
            percentile(latencies, 50),
            3
        ),
        "p95_retrieval_latency_ms": round(
            percentile(latencies, 95),
            3
        ),
        "mean_retrieval_latency_ms": round(
            mean(latencies),
            3
        ),
        "average_chunks_retrieved": round(
            mean(retrieved_counts),
            3
        ),
        "latency_scope": "ChromaDB query only; excludes embedding and generation"
    }


def create_default_assumptions():
    assumptions = {
        "self_hosted_monthly_cost_usd": {
            "100000": None,
            "1000000": None,
            "10000000": None
        },
        "pinecone_standard_monthly_cost_usd": {
            "100000": None,
            "1000000": None,
            "10000000": None
        },
        "qdrant_standard_monthly_cost_usd": {
            "100000": None,
            "1000000": None,
            "10000000": None
        },
        "notes": [
            "Enter monthly infrastructure costs for self-hosted ChromaDB.",
            "Enter actual managed database estimates from provider calculators.",
            "Do not treat null values as zero cost.",
            "Managed costs depend on dimensions, metadata, query volume, replicas, and resource configuration."
        ]
    }

    ASSUMPTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ASSUMPTIONS_PATH.write_text(
        json.dumps(assumptions, indent=2),
        encoding="utf-8"
    )

    return assumptions


def load_assumptions():
    if not ASSUMPTIONS_PATH.exists():
        return create_default_assumptions()

    return json.loads(
        ASSUMPTIONS_PATH.read_text(encoding="utf-8")
    )


def estimate_storage(vector_count):
    vector_bytes = (
        vector_count
        * EMBEDDING_DIMENSIONS
        * FLOAT_BYTES
    )

    metadata_bytes = (
        vector_count
        * METADATA_BYTES_PER_VECTOR
    )

    modeled_bytes = (
        vector_bytes + metadata_bytes
    ) * INDEX_OVERHEAD_MULTIPLIER

    return {
        "vector_bytes": vector_bytes,
        "metadata_bytes": metadata_bytes,
        "modeled_total_bytes": modeled_bytes,
        "modeled_total_gib": round(
            modeled_bytes / (1024 ** 3),
            3
        )
    }


def estimate_groq_monthly_cost():
    input_tokens = (
        MONTHLY_QUERIES
        * AVERAGE_INPUT_TOKENS_PER_QUERY
    )

    output_tokens = (
        MONTHLY_QUERIES
        * AVERAGE_OUTPUT_TOKENS_PER_QUERY
    )

    input_cost = (
        input_tokens
        / 1_000_000
        * GROQ_INPUT_PRICE_PER_MILLION
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * GROQ_OUTPUT_PRICE_PER_MILLION
    )

    return {
        "monthly_queries": MONTHLY_QUERIES,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 4),
        "output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(
            input_cost + output_cost,
            4
        )
    }


def build_cost_table(assumptions):
    rows = []

    for vector_count in VECTOR_COUNTS:
        storage = estimate_storage(vector_count)
        key = str(vector_count)

        rows.append({
            "vector_count": vector_count,
            "dimensions": EMBEDDING_DIMENSIONS,
            "modeled_storage_gib": storage["modeled_total_gib"],
            "self_hosted_monthly_usd": (
                assumptions["self_hosted_monthly_cost_usd"].get(key)
            ),
            "pinecone_standard_monthly_usd": (
                assumptions["pinecone_standard_monthly_cost_usd"].get(key)
            ),
            "qdrant_standard_monthly_usd": (
                assumptions["qdrant_standard_monthly_cost_usd"].get(key)
            )
        })

    return rows


def save_results(latency, storage_rows, groq_cost):
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    assumptions = load_assumptions()

    results = {
        "retrieval_latency": latency,
        "storage_assumptions": {
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "float_bytes": FLOAT_BYTES,
            "metadata_bytes_per_vector": METADATA_BYTES_PER_VECTOR,
            "index_overhead_multiplier": INDEX_OVERHEAD_MULTIPLIER,
            "note": "Storage is an estimate; actual ChromaDB disk usage must be measured."
        },
        "storage_estimates": storage_rows,
        "groq_generation_cost_estimate": groq_cost,
        "cost_comparison": build_cost_table(assumptions)
    }

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        fieldnames = list(results["cost_comparison"][0].keys())

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results["cost_comparison"])

    print(json.dumps(latency, indent=2))
    print(json.dumps(groq_cost, indent=2))
    print(f"Results saved to: {RESULTS_PATH}")
    print(f"Cost table saved to: {CSV_PATH}")
    print(f"Assumptions file: {ASSUMPTIONS_PATH}")


def main():
    latency = benchmark_retrieval()

    storage_rows = [
        {
            "vector_count": count,
            **estimate_storage(count)
        }
        for count in VECTOR_COUNTS
    ]

    groq_cost = estimate_groq_monthly_cost()

    save_results(
        latency,
        storage_rows,
        groq_cost
    )


if __name__ == "__main__":
    main()
