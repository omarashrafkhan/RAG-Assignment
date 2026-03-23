| Run | Chunking | Retrieval | Re-ranking | #Queries | Faithfulness | Relevancy | Note |
|---|---|---|---|---:|---:|---:|---|
| fixed__hybrid__rerank | fixed | Hybrid (BM25+Semantic+RRF) | Yes | 10 | 0.9800 | 0.7062 | full-test |
| fixed__semantic_only__no_rerank | fixed | Semantic-only | No | 10 | 0.9167 | 0.7143 | full-test |
| recursive__hybrid__rerank | recursive | Hybrid (BM25+Semantic+RRF) | Yes | 10 | 0.9000 | 0.7684 | full-test |
| recursive__semantic_only__no_rerank | recursive | Semantic-only | No | 10 | 0.9667 | 0.5339 | full-test |
| sentence__hybrid__rerank | sentence | Hybrid (BM25+Semantic+RRF) | Yes | 10 | 0.8167 | 0.7952 | full-test |
| sentence__semantic_only__no_rerank | sentence | Semantic-only | No | 10 | 0.9667 | 0.5589 | full-test |
