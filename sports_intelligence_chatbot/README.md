# Sports Intelligence Chatbot (RAG)

Offline-capable sports Q&A agent with autonomous query routing and multiple RAG strategies. Default sport examples assume Football/Soccer and Cricket, but you can load any sport content.

## Features

* Document ingestion (PDF, TXT, MD), chunking with overlap
* Sentence-Transformer embeddings + FAISS vector search
* Query classification: factual, comparative, analytical, creative, plus non-sport rejection
* Multi-strategy RAG: simple, hierarchical, multi-hop, adaptive thresholding
* Local LLM generation via Hugging Face (falls back if unavailable)
* Confidence scoring, citations, graceful fallbacks

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place sport documents under `data/sports_knowledge_base/`.

## Run demo

```bash
python demo.py
```

## Structure

```
src/
  document_processor.py
  vector_store.py
  query_classifier.py
  rag_strategies.py
  decision_engine.py
  sports_chatbot.py
  utils.py
```

## Notes

* Models are loaded from Hugging Face. First run requires internet; afterward runs offline.
* For laptops with ≤8GB RAM, prefer smaller LLMs (e.g., `mistralai/Mistral-7B-Instruct-v0.2` with 4-bit quant via bitsandbytes) or skip LLM to use extractive summaries.


