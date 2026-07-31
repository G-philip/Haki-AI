# Haki AI

**Making Kenyan law easier to understand through trustworthy AI.**

Haki AI is an AI-powered legal assistant built for Kenyan law. Rather
than relying on an AI model's memory, it uses a Retrieval-Augmented
Generation (RAG) pipeline to search a curated legal knowledge base
before generating every response. This helps ensure answers are grounded
in real legal sources instead of assumptions.

The project was created with one guiding principle:

> **If the system cannot support an answer with evidence, it should not
> guess.**

## Why Haki AI?

Legal information should be accurate, transparent, and easy to
understand. Haki AI bridges the gap between complex legal language and
everyday users by explaining legal provisions in plain English while
showing where the information came from.

Whether you're a student, researcher, advocate, or simply trying to
understand your rights, Haki AI aims to make legal information more
accessible.

## Key Features

-   🤖 Conversational AI for Kenyan legal questions
-   📚 Retrieval-Augmented Generation (RAG)
-   🔎 Semantic search over statutes and case law
-   ⚖️ Grounded answers backed by retrieved legal documents
-   📖 Automatic citations and source attribution
-   ✅ Validation of citations and quoted legal text
-   🚫 Hallucination-resistant answer pipeline
-   💬 Natural, easy-to-understand explanations
-   🔒 Refuses to fabricate answers when evidence is insufficient

## How It Works

1.  A user's question is converted into embeddings.
2.  The vector database retrieves the most relevant legal documents.
3.  Only the retrieved content is provided to the language model.
4.  Generated answers are validated against the retrieved sources.
5.  If an answer cannot be verified, Haki AI returns an honest response
    instead of making something up.

## Technology Stack

-   Python
-   Ollama
-   Llama 3.2
-   ChromaDB
-   Embedding models
-   Retrieval-Augmented Generation (RAG)

## Vision

We believe access to legal information should not depend on legal
training. Haki AI is designed to help people understand the law while
encouraging consultation with qualified advocates whenever professional
legal advice is needed.

Our long-term goal is to build a trusted legal intelligence platform for
Kenya that supports citizens, law firms, researchers, universities, and
public institutions.

## Roadmap

-   Web platform
-   Kiswahili support
-   Legal document analysis
-   Intelligent legal drafting
-   Advanced case law reasoning
-   Advocate workspace
-   Public API
-   Mobile application

## Disclaimer

Haki AI is intended for legal research and educational purposes. It is
**not** a substitute for professional legal advice. Users should consult
a qualified advocate for advice relating to their specific
circumstances.

------------------------------------------------------------------------

**Built with the belief that trustworthy AI begins with trustworthy
evidence.**
