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

1️⃣ Install Ollama**
Download from:  
👉 https://ollama.com/download

**OR (Linux/macOS):**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
model

2️⃣ Pull the Mistral Model
ollama pull mistral

3️⃣ Clone the Repository

4️⃣ Create Virtual Environment
python -m venv venv
then Activate it

5️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Running the Project
✔ Using Ollama (Recommended)

Runs fully locally using Mistral 7B:

python main.py --use-ollama

🛠 Tech Stack
- **Python 3.10+**
- **LangChain** (RAG pipeline)
- **ChromaDB** (vector store)
- **MiniLM-L6-v2** (embeddings)
- **Ollama Mistral 7B** (primary LLM)
- **Flan-T5-Small** (fallback LLM)


👤 Author

Alokik Gour
Kalpit Pvt Ltd — Intern Assignment
📧 Email: alokikgour29@gmail.com

🔗 GitHub: https://github.com/Alokik-29
