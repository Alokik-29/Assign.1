# 🧠 AmbedkarGPT — Intern Assignment (Kalpit Pvt Ltd)

This repository contains a **Retrieval-Augmented Generation (RAG)** prototype built for the **Kalpit Pvt Ltd internship assignment**.  
The system works fully **offline**, using **ChromaDB**, **HuggingFace embeddings**, and optionally **Ollama (Mistral 7B)** for LLM generation.

---

## 📌 What the Project Does

This RAG pipeline:

- Loads **speech.txt** (provided Ambedkar speech)
- Splits text into overlapping chunks
- Converts text into embeddings using  
  **sentence-transformers/all-MiniLM-L6-v2**
- Stores embeddings inside a **local ChromaDB** vector store
- Retrieves the most relevant context chunks
- Generates answers using:

### ✔ Preferred (Assignment Requirement)
**Ollama — Mistral 7B**  
Runs fully locally — no API keys, no cloud.

### ✔ Fallback (Optional Testing)
**HuggingFace Flan-T5-Small**  
Used automatically when `--use-ollama` is not passed.

---

## 📁 Files Overview

| File | Description |
|------|-------------|
| **main.py** | Main script (interactive Q&A CLI) |
| **speech.txt** | Provided Ambedkar speech text |
| **requirements.txt** | Dependency list |
| **chroma_db/** | Auto-generated vector store |

---

## 🚀 Quick Start (Local Machine with Ollama)

### **1️⃣ Install Ollama**
Download from:  
👉 https://ollama.com/download

**OR (Linux/macOS):**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
model


👤 Author

Alokik Gour
Kalpit Pvt Ltd — Intern Assignment
GitHub: https://github.com/Alokik-29
alokikgour29@gmail.com
