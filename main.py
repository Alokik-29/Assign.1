#!/usr/bin/env python3
"""
main.py

- Loads speech.txt (expects it in the same folder)
- Splits into chunks (chunk_size=500, overlap=50 like in your notebook)
- Creates HuggingFaceEmbeddings using sentence-transformers/all-MiniLM-L6-v2
- Stores embeddings in Chroma (persist_directory="./chroma_db")
- Uses Ollama (mistral) when --use-ollama is passed and Ollama Python wrapper is available
- Falls back to HuggingFace Flan-T5 small for quick testing (no API keys)

Usage:
    python main.py                # run with HF fallback (testing)
    python main.py --use-ollama   # run with Ollama (requires local Ollama + `ollama pull mistral`)
"""

import os
import argparse
import sys
from typing import Any, Dict

# Robust imports — work across different LangChain installs
# Document loader, embeddings, vector store
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except Exception:
    from langchain.document_loaders import TextLoader
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma

# Text splitter (different installs expose different modules)
try:
    from langchain_text_splitters import CharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitters import CharacterTextSplitter
    except Exception:
        from langchain.text_splitter import CharacterTextSplitter

# Prompt / chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# HF pipeline (fallback LLM)
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Try to import an Ollama wrapper safely (different envs expose different packages)
OLLAMA_AVAILABLE = False
OLLAMA_WRAPPER = None
try:
    from langchain_ollama import OllamaLLM  # type: ignore
    OLLAMA_AVAILABLE = True
    OLLAMA_WRAPPER = "langchain_ollama"
except Exception:
    try:
        from langchain.llms import Ollama  # type: ignore
        OLLAMA_AVAILABLE = True
        OLLAMA_WRAPPER = "langchain.llms"
    except Exception:
        OLLAMA_AVAILABLE = False
        OLLAMA_WRAPPER = None

# -----------------------
# Configuration
# -----------------------
SPEECH_FILE = "speech.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PERSIST_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_FALLBACK_MODEL = "google/flan-t5-small"
TOP_K = 3
# -----------------------

def ensure_speech_file_exists():
    if not os.path.exists(SPEECH_FILE):
        print(f"[ERROR] {SPEECH_FILE} not found in the current directory: {os.getcwd()}")
        print("Place the provided speech.txt in the repository root before running this script.")
        sys.exit(1)

def build_chunks() -> Any:
    """
    Load speech.txt and split into chunks using CharacterTextSplitter.
    """
    ensure_speech_file_exists()
    loader = TextLoader(SPEECH_FILE, encoding="utf-8")
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} document(s). First doc length: {len(documents[0].page_content)} chars")

    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    return chunks

def create_vectorstore(chunks: Any) -> Chroma:
    """
    Create embeddings and persist a Chroma vectorstore.
    """
    print("[INFO] Creating embeddings (this may download the model first time)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
    print("[INFO] Creating Chroma vectorstore and persisting to:", PERSIST_DIR)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    try:
        vectordb.persist()
    except Exception:
        # some versions persist on creation; ignore if not needed
        pass
    print(f"[INFO] Vectorstore created. Indexed ~{len(chunks)} vectors.")
    return vectordb

def make_hf_llm(model_name: str = HF_FALLBACK_MODEL):
    """
    Create a HuggingFace text2text-generation pipeline wrapped for LangChain.
    """
    print(f"[INFO] Loading HF fallback model: {model_name} (this may take ~minute).")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256, device=device)
    return HuggingFacePipeline(pipeline=pipe)

def make_ollama_llm():
    """
    Create an Ollama LLM wrapper. Expects Ollama installed locally and mistral pulled.
    """
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("Ollama wrapper not found in Python environment. Install langchain-ollama or use a LangChain version with Ollama support.")
    if OLLAMA_WRAPPER == "langchain_ollama":
        return OllamaLLM(model="mistral", temperature=0.2)  # type: ignore
    else:
        from langchain.llms import Ollama  # type: ignore
        return Ollama(model="mistral", temperature=0.2)  # type: ignore

def build_qa_chain(vectordb: Chroma, llm) -> RetrievalQA:
    """
    Construct the RetrievalQA chain using the LLM.
    """
    print("[INFO] Building RetrievalQA chain (chain_type='stuff', return_source_documents=True)")
    prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def _call_qa_chain(qa_chain, question: str):
    """
    Robust invocation helper that tries multiple call styles across different LangChain versions.
    Returns (answer:str, source_docs:list)
    """
    # 1) try invoke (preferred for multi-output)
    try:
        result = qa_chain.invoke({"query": question})
        if isinstance(result, dict):
            answer = result.get("result") or result.get("answer") or result.get("output_text") or ""
            source_docs = result.get("source_documents") or []
            return answer, source_docs
        elif isinstance(result, str):
            return result, []
    except Exception:
        pass

    # 2) try calling chain with dict (many versions support this)
    try:
        result = qa_chain({"query": question})
        if isinstance(result, dict):
            answer = result.get("result") or result.get("answer") or result.get("output_text") or ""
            source_docs = result.get("source_documents") or []
            return answer, source_docs
        elif isinstance(result, str):
            return result, []
    except Exception:
        pass

    # 3) try run (works only for single-output chains)
    try:
        result = qa_chain.run(question)
        if isinstance(result, str):
            return result, []
        elif isinstance(result, dict):
            answer = result.get("result") or result.get("answer") or ""
            source_docs = result.get("source_documents") or []
            return answer, source_docs
    except Exception:
        pass

    # fallback
    return "[ERROR] Could not invoke QA chain (see logs).", []

def interactive_loop(qa_chain: RetrievalQA):
    print("\n[READY] Ask a question (type 'exit' or Ctrl+C to quit).")
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        answer, source_docs = _call_qa_chain(qa_chain, q)
        print("\nAnswer:\n", answer)
        if source_docs:
            print("\n-- Source chunks used --")
            for i, d in enumerate(source_docs, start=1):
                snippet = (d.page_content[:400] + "...") if len(d.page_content) > 400 else d.page_content
                print(f"[{i}] {snippet}")
        print("-" * 60)

def main(use_ollama: bool):
    chunks = build_chunks()
    vectordb = create_vectorstore(chunks)

    if use_ollama:
        print("[INFO] Attempting to initialize Ollama LLM (local).")
        try:
            llm = make_ollama_llm()
            print("[INFO] Ollama initialized successfully.")
        except Exception as e:
            print(f"[WARN] Could not initialize Ollama: {e}")
            print("[INFO] Falling back to HuggingFace model for generation.")
            llm = make_hf_llm()
    else:
        llm = make_hf_llm()

    qa_chain = build_qa_chain(vectordb, llm)
    interactive_loop(qa_chain)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AmbedkarGPT - RAG Q&A (converted from Colab notebook)")
    parser.add_argument("--use-ollama", action="store_true", help="Use local Ollama (requires ollama binary and mistral model).")
    args = parser.parse_args()
    main(use_ollama=args.use_ollama)
