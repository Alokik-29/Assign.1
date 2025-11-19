import os
import json
import nltk
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Folder paths and config
CORPUS_DIR = "corpus"
TEST_DATA = "test_dataset.json"
RESULTS_JSON = "test_results.json"

# Three chunk settings to compare
CHUNK_CONFIGS = {
    "small": (250, 50),
    "medium": (550, 50),
    "large": (900, 100)
}

TOP_K = 3

# Embedding model for cosine similarity
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ROUGE evaluator
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# BLEU tokenizer
nltk.download("punkt", quiet=True)


# Load the dataset of questions
def load_test_dataset():
    with open(TEST_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["test_questions"]


# Build vectorstore for each chunk size
def build_vectorstore(chunk_size, overlap):
    docs = []

    for f in sorted(os.listdir(CORPUS_DIR)):
        if f.endswith(".txt"):
            loader = TextLoader(os.path.join(CORPUS_DIR, f), encoding="utf-8")
            docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        chunks,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory=None
    )
    return vectordb


# Retrieval metrics
def hit_rate(topk, gold_docs):
    return 1 if any(doc in topk for doc in gold_docs) else 0

def mrr(topk, gold_docs):
    for i, d in enumerate(topk, start=1):
        if d in gold_docs:
            return 1 / i
    return 0

def precision_at_k(topk, gold_docs):
    count = sum(1 for d in topk if d in gold_docs)
    return count / len(topk)


# Answer quality metrics
def rouge_l(ref, pred):
    return rouge.score(ref, pred)["rougeL"].fmeasure

def bleu(ref, pred):
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)

def cosine_sim(ref, pred):
    a = embedder.encode([ref])[0]
    b = embedder.encode([pred])[0]
    return cosine_similarity([a], [b])[0][0]


# Load local Mistral if available, otherwise Flan-T5
def load_llm():
    try:
        from langchain_ollama import OllamaLLM
        print("Using OLLAMA (mistral)")
        return OllamaLLM(model="mistral")
    except:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        print("Using HF fallback (flan-t5-small)")
        tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        pipe = pipeline("text2text-generation", model=model, tokenizer=tok)
        return lambda x: pipe(x)[0]["generated_text"]


# Full evaluation loop
def evaluate():
    questions = load_test_dataset()
    llm = load_llm()

    final_results = {}

    for name, (chunk_size, overlap) in CHUNK_CONFIGS.items():
        print(f"\nEvaluating chunk size: {name}")

        vectordb = build_vectorstore(chunk_size, overlap)
        retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

        config_results = []

        for q in tqdm(questions):
            question = q["question"]
            gold_docs = q["source_documents"]
            gt = q["ground_truth"]
            answerable = q["answerable"]

            # Retrieve documents
            retrieved = retriever.get_relevant_documents(question)
            topk = [d.metadata.get("source", "") for d in retrieved]

            # Retrieval metrics
            hr = hit_rate(topk, gold_docs)
            rr = mrr(topk, gold_docs)
            p = precision_at_k(topk, gold_docs)

            # Build context and generate answer
            context = "\n\n".join([d.page_content for d in retrieved])
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

            if answerable:
                pred = llm(prompt)
                rl = rouge_l(gt, pred)
                bl = bleu(gt, pred)
                cs = cosine_sim(gt, pred)
            else:
                pred = ""
                rl = bl = cs = 0

            config_results.append({
                "id": q["id"],
                "question": question,
                "gold_docs": gold_docs,
                "retrieval": {
                    "hit_rate": hr,
                    "mrr": rr,
                    "precision": p
                },
                "generated_answer": pred,
                "metrics": {
                    "rouge_l": rl,
                    "bleu": bl,
                    "cosine": cs
                }
            })

        final_results[name] = config_results

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("Saved:", RESULTS_JSON)


# Run evaluation
if __name__ == "__main__":
    evaluate()

