# AmbedkarGPT - Intern Assignment (Kalpit Pvt Ltd)

This repository contains a small Retrieval-Augmented Generation (RAG) prototype for the Kalpit intern assignment.

## What it does
- Loads `speech.txt` (provided excerpt)
- Splits text into chunks
- Creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Stores embeddings in a local ChromaDB vector store
- Retrieves context for user questions
- Generates answers using an LLM:
  - **Preferred (assignment):** Ollama with Mistral 7B (local)
  - **Fallback (for testing / Colab):** HuggingFace Flan-T5 model (no API keys)

## Files
- `main.py` — main script (interactive CLI)
- `speech.txt` — *please include the provided speech.txt in the repo root*
- `requirements.txt` — dependencies
- `.gitignore` — recommended ignore patterns

## Quick start (local machine — recommended for Ollama)
1. Install Ollama locally and pull the mistral model:
   ```bash
   # Install Ollama (example)
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull Mistral 7B
   ollama pull mistral
