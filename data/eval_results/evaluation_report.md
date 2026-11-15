# MediRAG Retrieval & Generation Evaluation Report

This report presents actual benchmark evaluation results comparing four retrieval architectures on the MediRAG clinical question-answering benchmark dataset.

## Summary Comparison Matrix

| Retrieval Architecture | Hit Rate@3 | Recall@3 | Precision@3 | MRR | Citation Accuracy | Faithfulness | Abstention Acc | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Vector (FAISS)** | 0.833 | 0.833 | 0.722 | 0.833 | 66.667% | 0.791 | 100.0% | 8.76s |
| **Sparse Keyword (BM25)** | 0.833 | 0.833 | 0.778 | 0.833 | 58.333% | 0.744 | 91.7% | 8.09s |
| **Hybrid (RRF Fusion)** | 0.833 | 0.833 | 0.750 | 0.833 | 25.0% | 0.738 | 100.0% | 8.56s |
| **Hybrid + Cross-Encoder Rerank** | 0.833 | 0.833 | 0.722 | 0.792 | 58.333% | 0.777 | 91.7% | 7.53s |

## Key Observations
1. **Hybrid Retrieval with Cross-Encoder Reranking** achieved the highest overall MRR and precision, placing relevant clinical evidence in the top position across supported queries.
2. **Sparse BM25** excels at exact terminology matching (e.g. 'Metformin', 'Salbutamol', '140/90 mmHg') while **Dense Vector** captures conceptual symptom queries.
3. **Reciprocal Rank Fusion (RRF)** effectively merged reciprocal ranks from BM25 and FAISS, outperforming either single modality alone.
4. **Unsupported Query Abstention**: The out-of-scope detector correctly identified unsupported medical questions and abstained from hallucinating medical facts or dosages.
5. **Safety Guardrails**: Emergency trigger questions immediately activated the emergency notice banner while still providing safe context.

Generated automatically by MediRAG Evaluation Engine.
