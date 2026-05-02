# 📄 AI Resume Analyzer (RAG-based)

## 🚀 Overview

An end-to-end **Resume Analyzer** built using a **Retrieval-Augmented Generation (RAG)** pipeline.
It allows users to upload a resume and ask natural language questions, returning accurate, context-aware answers.

---

## 🧠 Architecture

```text
PDF → Cleaning → Chunking → Embeddings → FAISS
                                      ↓
Query → Embedding → Retrieval → Top-k Chunks → LLM → Answer
```

---

## 🛠️ Tech Stack

* Python
* SentenceTransformers (embeddings)
* FAISS (vector search)
* Groq API (LLM)

---

## ⚙️ How It Works

1. Resume is parsed and cleaned
2. Text is split into chunks with overlap
3. Chunks are converted into embeddings
4. Stored in FAISS for similarity search
5. Query is embedded and matched
6. Top-k relevant chunks are retrieved
7. LLM generates answer using retrieved context

---

## ▶️ Usage

```bash
git clone <repo>
cd AI-Resume-Analyzer
pip install -r requirements.txt
python main.py
```

Add `.env`:

```text
GROQ_API_KEY=your_key_here
```

---

## 🎯 Key Features

* Semantic search over resume
* Context-aware LLM responses
* Modular pipeline (swap models easily)

## 📌 Example Queries

* What are the candidate’s skills?
* Summarize experience
* List projects

---

⭐ Star this repo if you found it useful!
