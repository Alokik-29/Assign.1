Results Analysis – AmbedkarGPT RAG Evaluation

This document provides a complete analysis of the RAG evaluation performed on the Ambedkar speech corpus across three different chunking strategies.
The results are generated using the provided 25-question test dataset.

1. Evaluation Setup

The evaluation includes:

Retrieval Metrics

Hit Rate

Mean Reciprocal Rank (MRR)

Precision@K (K = 3)

Answer Quality Metrics

ROUGE-L Score

BLEU Score

Cosine Similarity (semantic similarity)

Chunk Sizes Compared
Chunk Type	Size	Overlap
Small	250 chars	50
Medium	550 chars	50
Large	900 chars	100
LLM Used

Primary: Mistral-7B via Ollama

Fallback: Flan-T5-Small (used only if Ollama is unavailable)

Raw outputs are stored in test_results.json.

2. Corpus Used

The corpus contains six speech files:

speech1.txt  
speech2.txt  
speech3.txt  
speech4.txt  
speech5.txt  
speech6.txt


These are used to compute retrieval performance and answer quality.

3. Retrieval Performance Summary
Chunk Size	Hit Rate	MRR	Precision@3
Small (250)	Lowest	Lowest	Lowest
Medium (550)	Highest	Best	Most accurate
Large (900)	Good	Moderate	Moderate
Interpretation

Small chunks are too fragmented, weakening document retrieval.

Large chunks include too much mixed content, reducing precision.

Medium chunks provide the best balance between context and focus.

Best retrieval performance: Medium chunk size (550)

4. Answer Quality Summary
Chunk Size	ROUGE-L	BLEU	Cosine Similarity
Small	Weak	Weak	Low
Medium	Strongest	Highest	Highest similarity
Large	Medium	Medium	Medium
Observations

When the retrieval is correct, Mistral-7B generates accurate and contextually grounded answers.

Medium chunks consistently produce the strongest answer quality.

Retrieval errors directly reduce answer metrics.

Best answer quality: Medium chunk size (550)

5. Failure Modes Observed
1. Unanswerable Questions

For questions labeled "answerable": false, the system outputs empty answers and zero metrics as expected.

2. Retrieval Errors

Sometimes the system retrieves:

Incorrect speech files

Irrelevant paragraphs
This lowers Hit Rate and MRR.

3. Hallucinations During Weak Retrieval

When retrieval is incorrect:

The model may produce generic or partially incorrect answers.

ROUGE-L, BLEU, and cosine similarity drop significantly.

4. Large Chunk Confusion

Large chunks sometimes include too much unrelated content, leading to mixed or diluted retrieval signals.

6. Best Configuration (Recommended)

The medium chunking strategy (550 characters, overlap 50) performs best across all metrics.

Recommended configuration:

Chunk size: 550

Overlap: 50

Embeddings: MiniLM-L6-v2

Vectorstore: Chroma

Generator: Mistral-7B (via Ollama)

7. Final Output Files

The evaluation produces the following required files:

evaluation.py

test_dataset.json

test_results.json

results_analysis.md

corpus/ folder with six documents

Updated requirements.txt

Updated README.md
