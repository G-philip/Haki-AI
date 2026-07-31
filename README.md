# Haki AI

> **Evidence-first legal AI for Kenyan law.**

Haki AI is an open-source legal assistant that combines Retrieval-Augmented Generation (RAG), semantic search and grounding validation to help users understand Kenyan law using information retrieved from an indexed legal knowledge base.

Unlike a conventional chatbot, Haki AI is designed to retrieve relevant legal material before generating a response. The generated answer is then checked against the retrieved evidence to reduce unsupported statements and improve trustworthiness.

---

## Why Haki AI?

Legal AI should be transparent. Users deserve to know where information comes from and when the available evidence is insufficient.

Haki AI focuses on:

- Grounded legal responses
- Retrieval before generation
- Citation-aware answers
- Hallucination reduction
- Plain-language explanations

---

## Features

- Retrieval-Augmented Generation (RAG)
- Semantic search over legal documents
- Local LLM inference with Ollama
- Chroma vector database
- Legal document ingestion pipeline
- Citation and grounding validation
- Case law support
- Conversation management
- Modern Next.js frontend

---

## Architecture

```text
User Question
      │
      ▼
Embedding Model
      │
      ▼
Vector Search (Chroma)
      │
      ▼
Relevant Legal Documents
      │
      ▼
Grounding & Validation
      │
      ▼
LLM (Ollama / Llama)
      │
      ▼
Verified Response
```

---

## Repository Structure

```text
Haki-AI/
├── backend/
│   ├── logic/
│   ├── utils/
│   ├── data/
│   ├── chroma_db/
│   ├── logs/
│   └── *.py
├── frontend/
│   ├── app/
│   ├── components/
│   └── package.json
└── README.md
```

---

## Screenshots

Add screenshots later by placing them in:

```text
docs/screenshots/
```

Example:

```markdown
![Home](docs/screenshots/home.png)

![Chat](docs/screenshots/chat.png)

![Grounded Response](docs/screenshots/response.png)

![Retrieved Sources](docs/screenshots/retrieval.png)
```

---

## Tech Stack

- Python
- Next.js
- React
- Tailwind CSS
- Ollama
- ChromaDB

---

## Roadmap

- Improved legal reasoning
- Better citation visualisation
- Kiswahili support
- Public API
- Mobile client

---

## Disclaimer

Haki AI is intended for legal research and education. It is not a substitute for professional legal advice.

