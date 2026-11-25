## Discussion

### When would I switch to a managed vector database?

I chose ChromaDB with persistent local storage because it is simple to deploy, has no mandatory managed-service subscription, and provides sufficient functionality for a small RAG application.

I would consider switching to a managed vector database such as Pinecone or Qdrant Cloud when the application requires:

- High availability and reliable production uptime
- Horizontal scaling as the document collection grows
- Multiple application instances accessing the same index
- Managed backups, monitoring, and operational support
- Lower maintenance effort for infrastructure and database upgrades

The decision would depend on measured query traffic, storage requirements, operational constraints, and the managed provider's actual pricing. A managed database is not automatically cheaper; it trades infrastructure control and maintenance effort for a service cost.

### Was retrieval or generation the weak link?

In the initial evaluation, retrieval achieved perfect scores across Recall@K, Hit Rate@K, MRR, nDCG@K, and Context Precision@K. However, the benchmark contained only two answerable questions and one unanswerable question, so these results are preliminary.

The answer evaluation achieved a 66.67% pass rate and a mean score of 4.67/5. Faithfulness and abstention scored 5/5, while answer relevance and completeness were slightly lower.

Based on this small test, generation appears to be the more likely area for improvement, particularly answer completeness and relevance. This is not a definitive conclusion because the evaluation dataset is too small to establish a reliable weak link.

The next step is to evaluate a larger dataset with multi-chunk questions, cross-document questions, and unanswerable cases. This will help determine whether retrieval quality or generation quality limits the system under more realistic conditions.