# Scientific Literature & Document Intelligence

A Retrieval-Augmented Generation (RAG) system for grounded question-answering and summarization over scientific PDFs. Built for the GenAI Applications course, Milestone 2.

**Team:** Shashank Reddy Gillella, Meghansh Pochampally

---

## What This Does

Given a single scientific PDF, the system extracts clean text (including from scanned/OCR-damaged documents), indexes it for semantic search, and answers natural-language questions using only the document's actual content — flagging uncertainty rather than guessing when evidence is missing or weak. It also provides extractive and abstractive summarization.

The core problem this addresses: general-purpose LLM chatbots answer from training memory and can state outdated or fabricated details with full confidence. This system is scoped narrowly — one document, strictly grounded answers — to prove that mechanism works before scaling further.

---

## Architecture

```
PDF Upload
    │
    ▼
Extraction Fallback Chain
  PyMuPDF → pdfplumber → pdfminer → PyPDF2 → OCR (Tesseract)
  (best-scoring candidate selected)
    │
    ▼
Cleaning & Structure Parsing
  whitespace/OCR repair, title/abstract/intro detection, references excluded
    │
    ▼
Chunking
  sentence-preserving, section-aware, front-matter prioritized
    │
    ▼
FAISS Indexing
  MiniLM embeddings, stored per document hash
    │
    ▼
Retrieval & Reranking
  FAISS semantic + TF-IDF lexical, MMR diversity, context-budget selection
    │
    ▼
Grounded QA (Ollama, Llama 3.2)
  answers using retrieved context only, not model training knowledge
    │
    ▼
Guardrail Check
  scans for unsupported numeric/date/statistical claims → repair pass if triggered
```

---

## Tech Stack

| Layer | Tools Used |
|---|---|
| Extraction | PyMuPDF, pdfplumber, pdfminer.six, PyPDF2, pytesseract (OCR) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (local index) |
| Retrieval | FAISS semantic search + TF-IDF lexical matching (hybrid) |
| Generation | Ollama — Llama 3.2 (3B) for QA, Gemma 3 (1B) for summarization |
| Explainability | LIME, SHAP |
| Backend | Flask |
| Storage | SQLite (documents, chat history) |
| Frontend | HTML, CSS, JavaScript |

All components run locally — zero per-query API cost, no data leaves the machine. This was a deliberate choice to validate the core retrieval-and-grounding mechanism before investing in cloud infrastructure. See `Addendum_Slides.pptx` for the proposed ideal/scaled-up stack and trade-off discussion.

---

## Setup & Running

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Install and start [Ollama](https://ollama.com), then pull the required models:
   ```
   ollama pull llama3.2:3b
   ollama pull gemma3:1b
   ```
3. Run the app:
   ```
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in a browser.

---

## Repository Contents

- `app.py` — Flask routes (auth, chat, summarization, chat history)
- `scientific_rag.py` — core RAG pipeline (extraction, chunking, FAISS indexing, retrieval, grounded QA, guardrails)
- `config.py` — configuration
- `templates/`, `static/` — frontend
- `sample_documents/` — test PDFs used for evaluation (includes the RAG-Anything, LoRA, and Attention Is All You Need papers)
- `traces.log` — raw terminal trace output from real test runs (20-question and 24-question sets), documenting retrieval, generation, and guardrail behavior end to end
- `Milestone2_Technical_Deck.pptx` — Milestone 2 presentation
- `Addendum_Slides.pptx` — addendum covering architecture diagram, QA traces, and measured evaluation metrics
- `Milestone2_Metrics_Tracker.xlsx` — retrieval and hallucination-rate tracking workbook with auto-calculated summary metrics

---

## Evaluation Summary

Tested against a 20-question set on a real arXiv paper (RAG-Anything), comparing a no-retrieval baseline LLM against this grounded system:

- **Baseline LLM hallucination rate:** 100% (20/20) — fabricated authors, benchmarks, statistics, and publication venue when given no document context
- **Grounded system hallucination rate:** 0% (0/20) on the same questions
- **Guardrail repair pass:** did not trigger across 24 test questions (including 4 designed to bait fabrication) — the model consistently self-refused via prompt-level grounding before generation completed, rather than the guardrail catching a fabrication after generation

Full methodology, caveats, and raw data are in `Milestone2_Metrics_Tracker.xlsx` and `traces.log`.

---

## Known Limitations (Stated Honestly)

- **Single-document only** — no cross-document synthesis or contradiction detection across multiple papers yet
- **No agentic loop** — retrieval is single-pass (retrieve → generate → rule-based repair check), not an autonomous confidence-driven re-retrieval loop
- **Guardrail scope is narrow** — currently scans only for unsupported numeric/date/statistical tokens, not invented definitions or terminology (observed directly: a test query "what is RAG" produced an incorrect, ungrounded definition that the guardrail did not catch)
- **Chunk-level retrieval logging is incomplete** — current logs capture chunk *count* per query but not chunk *identity*, limiting true precision/recall computation to a proxy metric
- **Observed run-to-run non-determinism** — the same question against the same document produced different outputs (one fabricated, one correctly grounded) across separate sessions, likely due to LLM sampling variance

These are documented here rather than hidden, consistent with the addendum submitted for Milestone 2.

---

## Future Work

- Multi-document synthesis with cross-paper contradiction flagging
- Autonomous agentic re-retrieval loop based on generation confidence
- Chunk-identity logging for true retrieval precision/recall (Hit Rate@K, MRR)
- Migration path to managed infrastructure (hosted vector search, stronger hosted LLM, cloud deployment) — see `Addendum_Slides.pptx` for the proposed ideal stack
