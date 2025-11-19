# Results Analysis – AmbedkarGPT RAG Evaluation

Complete analysis of RAG evaluation on the Ambedkar speech corpus using three chunking strategies with a 25-question test dataset.

## Evaluation Setup

**Retrieval Metrics:**
- Hit Rate
- Mean Reciprocal Rank (MRR)
- Precision@K (K = 3)

**Answer Quality Metrics:**
- ROUGE-L Score
- BLEU Score
- Cosine Similarity

**Chunk Configurations:**

| Type | Size | Overlap |
|------|------|---------|
| Small | 250 chars | 50 |
| Medium | 550 chars | 50 |
| Large | 900 chars | 100 |

**Model:** Mistral-7B via Ollama (fallback: Flan-T5-Small if unavailable)

Raw outputs: `test_results.json`

## Corpus

Six speech files (speech1.txt through speech6.txt) used for evaluation.

## Retrieval Performance

| Chunk Size | Hit Rate | MRR | Precision@3 |
|------------|----------|-----|-------------|
| Small (250) | Lowest | Lowest | Lowest |
| Medium (550) | Highest | Best | Most accurate |
| Large (900) | Good | Moderate | Moderate |

Small chunks are too fragmented. Large chunks mix too much content together. Medium chunks strike the right balance.

**Winner:** Medium (550)

## Answer Quality

| Chunk Size | ROUGE-L | BLEU | Cosine Similarity |
|------------|---------|------|-------------------|
| Small | Weak | Weak | Low |
| Medium | Strongest | Highest | Highest |
| Large | Medium | Medium | Medium |

Mistral-7B performs well when retrieval is accurate. Medium chunks consistently yield better answers. Poor retrieval directly impacts answer quality.

**Winner:** Medium (550)

## Observed Issues

**Unanswerable Questions**  
Questions marked `"answerable": false` correctly return empty answers with zero metrics.

**Retrieval Errors**  
Wrong speech files or irrelevant paragraphs get retrieved sometimes, lowering Hit Rate and MRR.

**Hallucinations**  
Weak retrieval causes generic or incorrect outputs. All quality metrics drop when this happens.

**Large Chunk Problems**  
Too much unrelated content in large chunks dilutes retrieval signals.

## Recommended Setup

Medium chunking (550 chars, overlap 50) works best overall.

- Chunk size: 550
- Overlap: 50
- Embeddings: MiniLM-L6-v2
- Vectorstore: Chroma
- Generator: Mistral-7B (Ollama)

## Output Files

- evaluation.py
- test_dataset.json
- test_results.json
- results_analysis.md
- corpus/ (six documents)
- requirements.txt (updated)
- README.md (updated)
