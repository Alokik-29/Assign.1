import os
import json
import argparse
from tqdm import tqdm
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# CONFIG

CORPUS_DIR = "corpus"
TEST_FILE = "test_dataset.json"
RESULTS_FILE = "test_results.json"

CHUNK_SETTINGS = {
    "small": (250, 50),
    "medium": (550, 50),
    "large": (900, 100)
}

TOP_K = 3

# Metrics models
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
nltk.download("punkt", quiet=True)



# Load dataset

def load_test_data():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["test_questions"]



# Build vectorstore

def build_store(chunk_size, overlap):
    docs = []

    # Load files
    for fname in sorted(os.listdir(CORPUS_DIR)):
        if fname.endswith(".txt"):
            path = os.path.join(CORPUS_DIR, fname)
            loader = TextLoader(path, encoding="utf-8")
            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = fname
                docs.append(d)

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        chunks,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    return vectordb



# Metrics

def hit_rate(pred, gold):
    return 1.0 if any(x in gold for x in pred) else 0.0

def mrr(pred, gold):
    for i, doc in enumerate(pred, 1):
        if doc in gold:
            return 1 / i
    return 0.0

def precision_k(pred, gold):
    correct = sum(1 for d in pred if d in gold)
    return correct / len(pred)

def rouge_l(gt, ans):
    if not ans:
        return 0.0
    return rouge.score(gt, ans)["rougeL"].fmeasure

def bleu(gt, ans):
    if not ans:
        return 0.0
    smoothie = SmoothingFunction().method1
    return sentence_bleu([gt.split()], ans.split(), smoothing_function=smoothie)

def cosine(gt, ans):
    if not ans:
        return 0.0
    a = embedder.encode([gt])[0]
    b = embedder.encode([ans])[0]
    return float(cosine_similarity([a], [b])[0][0])



# Load LLM (simple)

def load_llm(use_ollama):
    if use_ollama:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model="mistral")
    else:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        pipe = pipeline("text2text-generation", model=model, tokenizer=tok)
        return lambda p: pipe(p)[0]["generated_text"]



# Main evaluation

def evaluate(use_ollama=False):
    data = load_test_data()
    llm = load_llm(use_ollama)

    final = {}

    for name, (chunk_size, overlap) in CHUNK_SETTINGS.items():
        print(f"\nRunning: {name} chunks...")
        vectordb = build_store(chunk_size, overlap)
        retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

        results = []

        for q in tqdm(data):
            question = q["question"]
            gold = q["source_documents"]
            gt = q["ground_truth"]
            answerable = q["answerable"]

            # retrieval
            retrieved = retriever.get_relevant_documents(question)
            topk = [d.metadata["source"] for d in retrieved]

            # retrieval metrics
            hr = hit_rate(topk, gold)
            rr = mrr(topk, gold)
            pk = precision_k(topk, gold)

            # answer generation
            if answerable:
                context = "\n\n".join([d.page_content for d in retrieved])
                prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                ans = llm(prompt)
                rl = rouge_l(gt, ans)
                bl = bleu(gt, ans)
                cs = cosine(gt, ans)
            else:
                ans = ""
                rl = bl = cs = 0.0

            results.append({
                "id": q["id"],
                "question": question,
                "gold_docs": gold,
                "topk": topk,
                "retrieval": {
                    "hit_rate": hr,
                    "mrr": rr,
                    "precision@k": pk
                },
                "answer": ans,
                "metrics": {
                    "rougeL": rl,
                    "bleu": bl,
                    "cosine": cs
                }
            })

        final[name] = results

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print("\n✔ Saved:", RESULTS_FILE)


# CLI Entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-ollama", action="store_true")
    args = parser.parse_args()

    evaluate(use_ollama=args.use_ollama)
