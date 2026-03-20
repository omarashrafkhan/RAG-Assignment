| Run | Chunking | Retrieval | Re-ranking | #Queries | Faithfulness | Relevancy | Note |
|---|---|---|---|---:|---:|---:|---|
| fixed__hybrid__rerank | fixed | Hybrid (BM25+Semantic+RRF) | Yes | 10 | 0.8750 | 0.8394 | full-test |
| fixed__semantic_only__no_rerank | fixed | Semantic-only | No | 10 | 0.7633 | 0.8017 | full-test |
| recursive__hybrid__rerank | recursive | Hybrid (BM25+Semantic+RRF) | Yes | 10 | 0.9467 | 0.7711 | full-test |
| recursive__semantic_only__no_rerank | recursive | Semantic-only | No | 10 | 0.7933 | 0.8131 | full-test |
| sentence__hybrid__rerank | sentence | Hybrid (BM25+Semantic+RRF) | Yes | 10 | 0.8933 | 0.8053 | full-test |
| sentence__semantic_only__no_rerank | sentence | Semantic-only | No | 10 | 0.8267 | 0.7238 | full-test |
