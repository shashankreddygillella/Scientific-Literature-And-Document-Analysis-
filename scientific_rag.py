# scientific_rag.py — Retrieval-augmented Q&A over scientific documents
# Primary retrieval: FAISS vector index (persistent local files)
# Fallback retrieval: in-memory TF-IDF
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence

import numpy as np
from ollama import chat as ollama_chat

try:
    from nltk.tokenize import sent_tokenize as _sent_tokenize
except ImportError:
    _sent_tokenize = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

OLLAMA_MODEL = "llama3.2:3b"
# RAG generation LLM (Ollama)
CHUNK_TARGET = 900
MAX_DOC_CHARS = 100_000
# Balanced tuning: keep latency controlled while preserving multi-clause coverage.
TOP_K_RETRIEVAL = 4
# Larger grounded context budget for multi-section QA.
MAX_CONTEXT_CHARS = 4500
FAISS_DIR = os.path.join(os.path.dirname(__file__), ".faiss_scientific_rag")
# Bump when chunking/retrieval selection changes so stale FAISS files are not reused.
RAG_INDEX_VERSION = "v9-ocr-structure-aware-front-matter"
_embedding_model = None
_index_lock = threading.Lock()
_indexed_doc_ids = set()
_faiss_import_failed = faiss is None
_tfidf_cache = {}
_tfidf_cache_lock = threading.Lock()
_SECTION_ALIASES = {
    "abstract": "abstract",
    "summary": "abstract",
    "executive summary": "abstract",
    "introduction": "introduction",
    "background": "background",
    "overview": "overview",
    "conclusion": "conclusion",
    "references": "references",
    "bibliography": "references",
    "related work": "related work",
    "methodology": "methodology",
    "methods": "methodology",
    "results": "results",
    "discussion": "discussion",
}
_SECTION_SCORE_OFFSETS = {
    "title": 0.40,
    "abstract": 0.35,
    "introduction": 0.20,
    "references": -0.50,
}
_FRONT_MATTER_KINDS = {"title", "abstract", "introduction"}
_SUMMARY_PRIORITY_KINDS = ("title", "abstract", "introduction", "conclusion")
_REFERENCE_HEADING_PATTERNS = (
    r"\breferences?\b",
    r"\bbibliograph(?:y|ies)\b",
    r"\bworks cited\b",
    r"\bcitations?\b",
)
_REFERENCE_CHUNK_PATTERNS = (
    r"\barxiv\b",
    r"\bdoi\b",
    r"\bproceedings of\b",
    r"\bet al\.\b",
    r"\bconference\b",
    r"\bjournal\b",
    r"\bvol\.\b",
    r"\bpp\.\b",
    r"\bassociation for computational linguistics\b",
    r"\bneural information processing systems\b",
    r"\bicml\b",
    r"\biclr\b",
    r"\bnips\b",
    r"\bneurips\b",
    r"\bacl\b",
    r"\bemnlp\b",
    r"\bnaacl\b",
    r"\beacl\b",
    r"\btransactions of\b",
    r"\bpreprint\b",
    r"\bpages?\b",
    r"\bno\.\b",
)


def _normalize_whitespace(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _repair_ocr_spaced_caps(text: str) -> str:
    s = text or ""
    if not s:
        return ""

    def _collapse(match: re.Match) -> str:
        return re.sub(r"\s+", "", match.group(0))

    prev = None
    while s != prev:
        prev = s
        s = re.sub(r"\b(?:[A-Z]\s+){2,}[A-Z]\b", _collapse, s)
        s = re.sub(r"\b([A-Z])\s+([A-Z][A-Z-]{2,})\b", r"\1\2", s)
        s = re.sub(r"\b([A-Z])\s+([A-Z][a-z]{2,})\b", r"\1\2", s)
        s = re.sub(r"\b([A-Z]{2,})\s*-\s*([A-Z]{2,})\b", r"\1-\2", s)
    return s


def _normalize_heading_line(text: str) -> str:
    return _normalize_whitespace(_repair_ocr_spaced_caps(text))


def _is_front_matter_kind(kind: Optional[str]) -> bool:
    return str(kind or "").strip().lower() in _FRONT_MATTER_KINDS


def _fix_broken_text(text: str) -> str:
    import re

    if not text:
        return ""

    # Add space between lowercase and uppercase letters
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    # Add space between letters and digits
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)

    # Recover missing spaces in long OCR-fused English tokens
    # so retrieval chunks are indexed with readable text.
    fused_pat = re.compile(r"\b[A-Za-z]{14,}\b")
    text = fused_pat.sub(lambda m: _split_fused_token(m.group(0)), text)

    # Repair OCR-spaced all-caps tokens like "I NTRODUCTION" and "L OW-RANK".
    text = "\n".join(_repair_ocr_spaced_caps(line) for line in text.splitlines())

    # Normalize horizontal spaces (preserve line boundaries for better chunking)
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _window_chunks(text: str, size: int, step: int) -> List[str]:
    chunks: List[str] = []
    i = 0
    while i < len(text):
        piece = text[i : i + size].strip()
        if piece:
            chunks.append(piece)
        i += step
    return chunks if chunks else [text[:size]]


def _safe_sentence_tokenize(text: str) -> List[str]:
    try:
        if _sent_tokenize is not None:
            return [s.strip() for s in _sent_tokenize(text) if s and s.strip()]
        raise RuntimeError("nltk optional")
    except Exception:
        return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]


def _looks_like_table_noise(text: str) -> bool:
    s = _normalize_whitespace(text)
    if not s:
        return True
    if re.search(r"^[\d\s|:_./\\-]+$", s):
        return True

    alpha = len(re.findall(r"[A-Za-z]", s))
    digits = len(re.findall(r"\d", s))
    pipes = s.count("|")
    tabs = s.count("\t")
    words = re.findall(r"[A-Za-z]{2,}", s)
    alpha_density = alpha / max(len(s), 1)
    digit_density = digits / max(len(s), 1)

    if len(words) < 4 and digit_density > 0.22:
        return True
    if alpha_density < 0.28 and digit_density > 0.16:
        return True
    if (pipes >= 2 or tabs >= 2) and alpha_density < 0.45:
        return True
    if re.search(r"(?:\b\d+\b[\s|]{2,}){3,}", s):
        return True
    return False


def _looks_like_front_matter_meta(text: str) -> bool:
    line = _normalize_heading_line(text)
    if not line:
        return False
    lower = line.lower()
    return bool(
        re.search(
            r"\b(?:published|conference paper|accepted at|appearing in|to appear|under review|preprint|camera-ready)\b",
            lower,
        )
    )


def _is_low_value_chunk(text: str) -> bool:
    s = _normalize_whitespace(text)
    if not s or len(s) < 35:
        return True
    if _looks_like_table_noise(s):
        return True
    if len(re.findall(r"[A-Za-z]", s)) < 18 and len(s) < 120:
        return True
    return False


def _normalize_heading_key(text: str) -> str:
    key = _normalize_heading_line(text)
    key = re.sub(r"^(?:(?:section|chapter)\s+)?(?:\d+(?:\.\d+)*|[IVXivx]+)[.)-]?\s+", "", key)
    key = re.sub(r"\s+", " ", key).strip(" :.-").lower()
    return key


def _classify_heading(text: str) -> Optional[str]:
    line = _normalize_heading_line(text)
    if not line:
        return None
    key = _normalize_heading_key(line)
    line_lower = line.strip(" :.-").lower()
    if key in _SECTION_ALIASES:
        return _SECTION_ALIASES[key]
    if re.fullmatch(r"(?:section|chapter)\s+\d+[a-z]?", line_lower):
        return "section"
    if re.fullmatch(r"(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.)-]?\s+[a-z].*", line_lower):
        return "section"
    return None


def _is_heading_candidate(text: str) -> bool:
    line = _normalize_heading_line(text)
    if not line or len(line) > 180:
        return False
    if line.endswith((".", ";", "?")):
        return False
    words = line.split()
    if len(words) > 20:
        return False
    if _classify_heading(line):
        return True
    letters = re.findall(r"[A-Za-z]", line)
    if not letters:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(len(letters), 1)
    titleish = sum(1 for w in words if w[:1].isupper()) / max(len(words), 1)
    digit_count = len(re.findall(r"\d", line))
    return (upper_ratio > 0.68 or titleish > 0.72) and digit_count <= 8


def _looks_like_title(text: str) -> bool:
    line = _normalize_heading_line(text)
    if not line or len(line) > 260:
        return False
    if line.endswith((".", ";", "?")):
        return False
    words = line.split()
    if not 3 <= len(words) <= 32:
        return False
    if _classify_heading(line) is not None or _looks_like_reference_chunk(line):
        return False
    if _looks_like_front_matter_meta(line):
        return True
    letters = re.findall(r"[A-Za-z]", line)
    if len(letters) < 12:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(len(letters), 1)
    titleish = sum(1 for w in words if w[:1].isupper()) / max(len(words), 1)
    sentencey = bool(re.search(r"\b(?:we|this|our|that|which|using|propose|show|present)\b", line.lower()))
    return (upper_ratio > 0.34 or titleish > 0.55) and not sentencey


def _split_oversized_sentence(sentence: str, max_chars: int) -> List[str]:
    s = _normalize_whitespace(sentence)
    if len(s) <= max_chars:
        return [s]

    pieces: List[str] = []
    buf = ""
    clauses = re.split(r"(?<=[;:])\s+|(?<=,)\s+(?=(?:which|that|who|when|where|while|and|or)\b)", s)
    clauses = [c.strip() for c in clauses if c and c.strip()]
    if not clauses:
        clauses = [s]

    for clause in clauses:
        addition = clause if not buf else f"{buf} {clause}"
        if len(addition) <= max_chars:
            buf = addition
            continue
        if buf:
            pieces.append(buf.strip())
            buf = ""
        if len(clause) <= max_chars:
            buf = clause
            continue
        words = clause.split()
        word_buf: List[str] = []
        cur = 0
        for word in words:
            add_len = len(word) + (1 if word_buf else 0)
            if word_buf and cur + add_len > max_chars:
                pieces.append(" ".join(word_buf).strip())
                word_buf = [word]
                cur = len(word)
            else:
                word_buf.append(word)
                cur += add_len
        if word_buf:
            pieces.append(" ".join(word_buf).strip())
    if buf:
        pieces.append(buf.strip())
    return [p for p in pieces if p]


def _split_text_preserving_sentences(text: str, max_chars: int) -> List[str]:
    s = _normalize_whitespace(text)
    if not s:
        return []
    if len(s) <= max_chars:
        return [s]

    sentences = _safe_sentence_tokenize(s)
    if not sentences:
        return _window_chunks(s, max_chars, max(200, max_chars // 3))

    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    for sentence in sentences:
        for part in _split_oversized_sentence(sentence, max_chars):
            add_len = len(part) + (1 if buf else 0)
            if buf and cur + add_len > max_chars:
                chunks.append(" ".join(buf).strip())
                buf = []
                cur = 0
            if len(part) > max_chars:
                for window in _window_chunks(part, max_chars, max_chars):
                    ww = _normalize_whitespace(window)
                    if ww:
                        chunks.append(ww)
                continue
            buf.append(part)
            cur += add_len
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]


def _prefix_chunk_text(kind: str, heading: Optional[str], body: str) -> str:
    clean_body = _normalize_whitespace(body)
    if not clean_body:
        return ""
    if kind == "title":
        return f"Title: {clean_body}"
    if kind == "abstract":
        return f"Abstract: {clean_body}"
    if kind == "introduction":
        return f"Introduction: {clean_body}"
    if heading:
        heading_text = _normalize_whitespace(heading)
        if heading_text and not clean_body.lower().startswith(heading_text.lower()):
            return f"{heading_text}\n{clean_body}"
    return clean_body


def _ocr_heading_pattern(label: str) -> str:
    parts = []
    for word in label.split():
        if word.isalpha():
            parts.append(r"\s*".join(re.escape(ch) for ch in word))
        else:
            parts.append(re.escape(word))
    return r"\s+".join(parts)


def _extract_inline_heading_block(text: str):
    line = _normalize_heading_line(text)
    if not line or len(line) > 320:
        return None

    unique_labels = sorted(set(_SECTION_ALIASES.keys()), key=len, reverse=True)
    for label in unique_labels:
        pat = _ocr_heading_pattern(label)
        match = re.match(
            rf"^(?P<prefix>(?:(?:section|chapter)\s+)?(?:\d+(?:\.\d+)*|[IVXivx]+)[.)-]?\s+)?"
            rf"(?P<head>{pat})(?:\s*[:.-]\s*|\s+)(?P<body>.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        body = _normalize_whitespace(match.group("body"))
        if len(body) < 20:
            continue
        heading = _normalize_heading_line(f"{match.group('prefix') or ''}{match.group('head')}")
        return heading, _SECTION_ALIASES[label], body
    return None


def _extract_title_from_block(block: str):
    lines = [_normalize_heading_line(line) for line in block.splitlines() if _normalize_whitespace(line)]
    if not lines:
        return None, block

    candidates = []
    for idx, line in enumerate(lines[:4]):
        if len(line) <= 260:
            candidates.append((line, idx + 1, idx))
    if len(lines) >= 2 and not _looks_like_front_matter_meta(lines[0]):
        combined = _normalize_heading_line(" ".join(lines[:2]))
        if len(combined) <= 260:
            candidates.insert(0, (combined, 2, 0))

    best = None
    best_score = -10**9
    for candidate, used_lines, start_idx in candidates:
        if not _looks_like_title(candidate):
            continue
        letters = re.findall(r"[A-Za-z]", candidate)
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(len(letters), 1)
        score = int(start_idx == 0) * 5 + int(not _looks_like_front_matter_meta(candidate)) * 4 + int(upper_ratio > 0.55) * 3 + len(candidate)
        if score > best_score:
            best = (candidate, used_lines)
            best_score = score

    if best is not None:
        candidate, used_lines = best
        remainder_lines = [line for idx, line in enumerate(lines) if idx >= used_lines or idx < used_lines - 1 and _looks_like_front_matter_meta(lines[idx])]
        if used_lines == 1 and len(lines) > 1 and _looks_like_front_matter_meta(lines[1]):
            remainder_lines = lines[1:]
        remainder = "\n".join(remainder_lines).strip()
        return candidate, remainder
    return None, block


def _find_embedded_heading_index(lines: Sequence[str]) -> int:
    for idx, line in enumerate(lines):
        normalized = _normalize_heading_line(line)
        if not normalized:
            continue
        if _extract_inline_heading_block(normalized):
            return idx
        if _is_heading_candidate(normalized) and _classify_heading(normalized):
            return idx
    return -1


def _build_chunk_records(
    text: str,
    max_chars: int = CHUNK_TARGET,
) -> List[Dict[str, object]]:
    doc_text = text or ""
    if not doc_text.strip():
        return []

    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n+", doc_text) if b and b.strip()]
    if not raw_blocks:
        raw_blocks = [doc_text.strip()]

    title = None
    idx = 0
    while idx < len(raw_blocks):
        block = _fix_broken_text(raw_blocks[idx])
        block = re.sub(r"\n{3,}", "\n\n", block).strip()
        if not block or _looks_like_table_noise(block):
            idx += 1
            continue
        title_candidate, remainder = _extract_title_from_block(block)
        if title_candidate:
            title = title_candidate
            if remainder:
                raw_blocks[idx] = remainder
            else:
                idx += 1
        break

    records: List[Dict[str, object]] = []
    order = 0
    if title:
        records.append({"text": f"Title: {title}", "kind": "title", "order": order})
        order += 1

    current_heading: Optional[str] = None
    current_kind = "body"
    current_paragraphs: List[str] = []

    def flush_current() -> None:
        nonlocal order, current_paragraphs
        if not current_paragraphs:
            return
        paragraphs = [_normalize_whitespace(p) for p in current_paragraphs if _normalize_whitespace(p)]
        current_paragraphs = []
        if not paragraphs:
            return

        section_text = "\n\n".join(paragraphs)
        clause_blocks = _split_clause_blocks(section_text) if current_kind not in {"abstract", "title"} else []
        units = clause_blocks if len(clause_blocks) >= 3 else paragraphs
        preserve_kind = _is_front_matter_kind(current_kind)

        for unit in units:
            if not preserve_kind and _is_low_value_chunk(unit):
                continue
            for part in _split_text_preserving_sentences(unit, max_chars):
                chunk_text = _prefix_chunk_text(current_kind, current_heading, part)
                if not preserve_kind and _is_low_value_chunk(chunk_text):
                    continue
                records.append(
                    {
                        "text": chunk_text,
                        "kind": current_kind,
                        "heading": current_heading,
                        "order": order,
                    }
                )
                order += 1

    for raw in raw_blocks[idx:]:
        block = _fix_broken_text(raw)
        block = re.sub(r"\n(?=[a-z])", " ", block)
        block = re.sub(r"\n{3,}", "\n\n", block).strip()
        if not block:
            continue

        block_lines = [line.strip() for line in block.splitlines() if line and line.strip()]
        if block_lines:
            heading_start = _find_embedded_heading_index(block_lines)
            if heading_start > 0:
                preamble_lines = [
                    line for line in block_lines[:heading_start]
                    if not _looks_like_table_noise(line) and not _looks_like_front_matter_meta(line)
                ]
                preamble = "\n".join(preamble_lines).strip()
                if preamble:
                    current_paragraphs.append(preamble)
                block_lines = block_lines[heading_start:]

            first_line = _normalize_heading_line(block_lines[0])
            trailing_lines = [line for line in block_lines[1:] if not _looks_like_table_noise(line)]
            inline_heading = _extract_inline_heading_block(first_line)
            if inline_heading:
                heading_text, heading_kind, inline_body = inline_heading
                flush_current()
                current_heading = heading_text
                current_kind = heading_kind
                body_parts = [inline_body] + trailing_lines
                if body_parts:
                    current_paragraphs.append("\n".join(body_parts).strip())
                continue
            heading_kind = _classify_heading(first_line) if _is_heading_candidate(first_line) else None
            if heading_kind:
                flush_current()
                current_heading = _normalize_heading_line(first_line)
                current_kind = heading_kind
                if trailing_lines:
                    current_paragraphs.append("\n".join(trailing_lines).strip())
                continue
            block = "\n".join(line for line in block_lines if not _looks_like_table_noise(line)).strip()

        if not block or _looks_like_table_noise(block):
            continue
        inline_heading = _extract_inline_heading_block(block)
        if inline_heading:
            heading_text, heading_kind, inline_body = inline_heading
            flush_current()
            current_heading = heading_text
            current_kind = heading_kind
            current_paragraphs.append(inline_body)
            continue
        heading_kind = _classify_heading(block) if _is_heading_candidate(block) else None
        if heading_kind:
            flush_current()
            current_heading = _normalize_heading_line(block)
            current_kind = heading_kind
            continue
        current_paragraphs.append(block)

    flush_current()

    if not records:
        for part in _split_text_preserving_sentences(_fix_broken_text(doc_text), max_chars):
            if not _is_low_value_chunk(part):
                records.append({"text": part, "kind": "body", "order": order})
                order += 1

    deduped: List[Dict[str, object]] = []
    seen = set()
    for record in records:
        txt = _normalize_whitespace(str(record.get("text", "")))
        kind = _record_kind(record)
        dedupe_key = (kind, txt) if _is_front_matter_kind(kind) else ("text", txt)
        if not txt or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        record["text"] = txt
        deduped.append(record)

    return deduped


def _record_text(record: Dict[str, object]) -> str:
    return _normalize_whitespace(str(record.get("text", "")))


def _record_kind(record: Dict[str, object]) -> str:
    return str(record.get("kind", "") or "").strip().lower()


def _record_heading(record: Dict[str, object]) -> str:
    return _normalize_whitespace(str(record.get("heading", "") or ""))


def _is_reference_heading(heading: Optional[str]) -> bool:
    line = _normalize_whitespace(heading or "")
    if not line:
        return False
    key = _normalize_heading_key(line)
    return any(re.search(pattern, key) for pattern in _REFERENCE_HEADING_PATTERNS)


def _looks_like_reference_chunk(text: str) -> bool:
    s = _normalize_whitespace(text).lower()
    if not s:
        return False
    hit_count = sum(1 for pattern in _REFERENCE_CHUNK_PATTERNS if re.search(pattern, s))
    year_hits = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", s))
    author_initial_hits = len(re.findall(r"\b[A-Z][a-z]+,\s*(?:[A-Z]\.\s*){1,3}", text))
    doi_like = bool(re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, flags=re.IGNORECASE))
    citation_punct_hits = len(re.findall(r"[.;]\s+[A-Z][a-z]+,\s*(?:[A-Z]\.\s*){1,3}", text))

    if hit_count >= 2:
        return True
    # Extra guard for citation-dense lines that OCR often flattens into body-looking chunks.
    if hit_count >= 1 and year_hits >= 2:
        return True
    if hit_count >= 1 and author_initial_hits >= 2:
        return True
    if doi_like and (year_hits >= 1 or author_initial_hits >= 1):
        return True
    if year_hits >= 2 and author_initial_hits >= 2:
        return True
    if citation_punct_hits >= 2 and year_hits >= 1:
        return True
    return False


def _is_reference_record(record: Dict[str, object]) -> bool:
    if _record_kind(record) == "references":
        return True
    if _is_reference_heading(_record_heading(record)):
        return True
    text = _record_text(record)
    text_head = text.splitlines()[0] if text else ""
    if _is_reference_heading(text_head):
        return True
    return _looks_like_reference_chunk(text)


def _is_clause_like_record(record: Dict[str, object]) -> bool:
    text = _record_text(record)
    heading = _record_heading(record)
    kind = _record_kind(record)
    if re.search(r"(?m)^\d{1,2}(?:\.\d+)*\.\s", text):
        return True
    if re.search(r"(?m)^\d{1,2}(?:\.\d+)*\s+[A-Za-z]", text):
        return True
    if kind == "section":
        return True
    if heading and re.match(r"^(?:section\s+\d+[a-z]?|\d+(?:\.\d+)*|[ivxlcdm]+[.)-])\b", heading.lower()):
        return True
    return False


def _classify_query_intent(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "factoid"
    if _is_document_summary_question(q):
        return "summary"
    clause_patterns = (
        r"\bclause\b",
        r"\bsection\b",
        r"\barticle\b",
        r"\bprovision\b",
        r"\bterm[s]?\b",
        r"\bobligation[s]?\b",
        r"\bliabilit(?:y|ies)\b",
        r"\bindemn(?:ity|ify|ification)\b",
        r"\btermination\b",
        r"\bright[s]?\b",
        r"\bdut(?:y|ies)\b",
        r"\bbreach\b",
        r"\bpayment\b",
        r"\bmaintenance\b",
        r"\bcustody\b",
        r"\b\d{1,2}(?:\.\d+)*\b",
    )
    if any(re.search(pattern, q) for pattern in clause_patterns):
        return "clause"
    return "factoid"


def _is_document_summary_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    patterns = [
        r"\bsummar(?:y|ize)\b",
        r"\boverview\b",
        r"\bpurpose\b",
        r"\bgist\b",
        r"\bmain (?:idea|point|argument|topic)\b",
        r"\bwhat is (?:this|the) (?:document|paper|article|agreement) about\b",
        r"\bwhat is (?:the )?purpose of (?:this|the) (?:document|paper|article|agreement)\b",
        r"\bwhat is this paper about\b",
        r"\bexplain (?:this|the) (?:document|paper|article|agreement)\b",
        r"\bkey (?:points|findings|takeaways)\b",
        r"\babstract\b",
    ]
    return any(re.search(pattern, q) for pattern in patterns)


def _priority_lead_chunk_records(question: str, chunk_records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    if not _is_document_summary_question(question):
        return []

    selected: List[Dict[str, object]] = []
    seen = set()
    kind_limits = {"title": 1, "abstract": 2, "introduction": 1, "conclusion": 1}
    for kind in _SUMMARY_PRIORITY_KINDS:
        taken = 0
        for record in chunk_records:
            if _record_kind(record) != kind:
                continue
            text = _record_text(record)
            if not text or text in seen:
                continue
            selected.append(dict(record))
            seen.add(text)
            taken += 1
            if taken >= kind_limits[kind]:
                break
    if not any(_record_kind(record) == "introduction" for record in chunk_records):
        for record in chunk_records:
            kind = _record_kind(record)
            text = _record_text(record)
            if kind in {"title", "abstract", "references"} or not text or text in seen:
                continue
            selected.append(dict(record))
            break
    return selected


def _priority_lead_chunks(question: str, chunk_records: Sequence[Dict[str, object]]) -> List[str]:
    return [_record_text(record) for record in _priority_lead_chunk_records(question, chunk_records)]


def _split_clause_blocks(text: str) -> List[str]:
    """
    Split documents into clause-like blocks when numbered sections exist
    (e.g., '1.', '2.', ...). This improves retrieval for clause-specific questions.
    """
    t = text or ""
    if not t:
        return []
    # Normalize obvious OCR spacing around clause markers first:
    # "1 . The" -> "1. The"
    t = re.sub(r"(?<!\d)(\d{1,2})\s*\.\s*", r"\1. ", t)

    # Split before numbered clauses even when they appear mid-line, e.g.
    # "... otherwise. 2. The husband ..."
    # "... rights or otherwise 2. The husband ..."
    boundary = re.compile(r"(?:(?<=^)|(?<=\n)|(?<=[.;:]))\s*(?=\d{1,2}\.\s)")
    parts = boundary.split(t)
    parts = [p.strip() for p in parts if p and p.strip()]

    # Keep only numbered clause chunks when structured section markers are present.
    clause_pat = re.compile(r"^\d{1,2}\.\s")
    blocks = [p for p in parts if clause_pat.match(p)]

    # If no numbered clauses are detected, fall back to previous behavior.
    if not blocks:
        return [p for p in parts if len(p) >= 40]

    # Ensure one logical clause per chunk by cutting off at next inline marker if any survived.
    cleaned: List[str] = []
    seen = set()
    for b in blocks:
        m = re.search(r"\s(?=\d{1,2}\.\s)", b[3:])
        chunk = b[: m.start() + 3].strip() if m else b.strip()
        if chunk and chunk not in seen:
            seen.add(chunk)
            cleaned.append(chunk)
    return cleaned


def chunk_document(text: str, max_chars: int = CHUNK_TARGET) -> List[str]:
    """Split document into overlapping segments for retrieval."""
    text = _normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Clause-aware path for structured documents:
    # one clause = one chunk (no cross-clause merging).
    clause_blocks = _split_clause_blocks(text)
    if len(clause_blocks) >= 4:
        out: List[str] = []
        seen = set()
        for block in clause_blocks:
            cc = block.strip()
            if not cc:
                continue
            # Keep clause boundaries strict. If a single clause is very large,
            # split only within that clause (no mixing with other clauses).
            if len(cc) > max_chars:
                for part in _window_chunks(cc, max_chars, max_chars):
                    pp = part.strip()
                    if pp and pp not in seen:
                        seen.add(pp)
                        out.append(pp)
                continue
            if cc not in seen:
                seen.add(cc)
                out.append(cc)
        if out:
            return out

    try:
        if _sent_tokenize is not None:
            sents = [s.strip() for s in _sent_tokenize(text) if s and s.strip()]
        else:
            raise RuntimeError("nltk optional")
    except Exception:
        sents = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]

    if not sents:
        return _window_chunks(text, max_chars, max_chars - 200)

    avg = len(text) / max(len(sents), 1)
    if len(sents) == 1 or avg > max_chars * 0.85:
        return _window_chunks(text, max_chars, max(200, max_chars // 3))

    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    overlap_tail = 2

    for s in sents:
        add_len = len(s) + (1 if buf else 0)
        if cur + add_len > max_chars and buf:
            chunks.append(" ".join(buf).strip())
            buf = buf[-overlap_tail:] if len(buf) > overlap_tail else buf[-1:]
            cur = sum(len(x) + 1 for x in buf)
        buf.append(s)
        cur += add_len
    if buf:
        chunks.append(" ".join(buf).strip())

    out: List[str] = []
    seen = set()
    for c in chunks:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out if out else [text[:max_chars]]


def chunk_document(text: str, max_chars: int = CHUNK_TARGET) -> List[str]:
    """Split document into semantic, sentence-preserving retrieval chunks."""
    chunk_records = _build_chunk_records(text, max_chars=max_chars)
    return [str(record["text"]) for record in chunk_records]


def _compute_tfidf_similarities(question: str, chunks: Sequence[str]) -> np.ndarray:
    chunks = [_normalize_whitespace(str(c)) for c in chunks if c and _normalize_whitespace(str(c))]
    if not chunks or not (question or "").strip():
        return np.zeros(len(chunks), dtype=np.float32)

    cache_key = hashlib.sha256(
        (RAG_INDEX_VERSION + "||" + "\n\n".join(chunks)).encode("utf-8", errors="ignore")
    ).hexdigest()
    with _tfidf_cache_lock:
        cached = _tfidf_cache.get(cache_key)
    if cached is None:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
        )
        try:
            doc_mat = vectorizer.fit_transform(chunks)
        except ValueError:
            return np.zeros(len(chunks), dtype=np.float32)
        with _tfidf_cache_lock:
            if len(_tfidf_cache) >= 6:
                _tfidf_cache.pop(next(iter(_tfidf_cache)))
            _tfidf_cache[cache_key] = (vectorizer, doc_mat)
    else:
        vectorizer, doc_mat = cached
    try:
        q_mat = vectorizer.transform([(question or "").strip()])
        sims = cosine_similarity(q_mat, doc_mat).flatten()
    except ValueError:
        return np.zeros(len(chunks), dtype=np.float32)
    return np.asarray(sims, dtype=np.float32)


def _normalize_score_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi > lo:
        return (arr - lo) / (hi - lo)
    if hi > 0:
        return np.ones_like(arr)
    return np.zeros_like(arr)


def _section_score_offset(record: Dict[str, object]) -> float:
    if _is_reference_record(record):
        return _SECTION_SCORE_OFFSETS["references"]
    return float(_SECTION_SCORE_OFFSETS.get(_record_kind(record), 0.0))


def _intent_score_offset(intent: str, record: Dict[str, object]) -> float:
    kind = _record_kind(record)
    if intent == "summary":
        if kind == "conclusion":
            return 0.18
        if kind in {"overview", "background"}:
            return 0.08
        return 0.0
    if intent == "clause":
        if _is_clause_like_record(record):
            return 0.18
        if kind == "section":
            return 0.10
    return 0.0


def _select_compact_context_record_items(
    records: Sequence[Dict[str, object]],
    max_chunks: int,
    max_context_chars: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen = set()
    total = 0
    for record in records:
        text = _record_text(record)
        if not text or text in seen:
            continue
        if total + len(text) > max_context_chars and out:
            break
        out.append(record)
        seen.add(text)
        total += len(text)
        if len(out) >= max_chunks or total >= max_context_chars:
            break
    return out


def retrieve_relevant_chunks(
    question: str,
    chunks: Sequence[str],
    top_k_scan: int = TOP_K_RETRIEVAL,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> List[str]:
    """Fallback lexical retrieval with light section-aware reranking."""
    records = [{"text": c, "kind": "body", "order": i} for i, c in enumerate(chunks or [])]
    records = [r for r in records if _record_text(r)]
    if not records:
        return []
    q = (question or "").strip()
    if not q:
        return [_record_text(r) for r in records[: min(3, len(records))]]

    tfidf_scores = _compute_tfidf_similarities(q, [_record_text(r) for r in records])
    tfidf_norm = _normalize_score_array(tfidf_scores)
    intent = _classify_query_intent(q)

    ranked: List[Dict[str, object]] = []
    for idx, record in enumerate(records):
        scored = dict(record)
        scored["tfidf_score"] = float(tfidf_scores[idx]) if idx < len(tfidf_scores) else 0.0
        scored["final_score"] = (
            0.85 * float(tfidf_norm[idx] if idx < len(tfidf_norm) else 0.0)
            + _section_score_offset(record)
            + _intent_score_offset(intent, record)
        )
        ranked.append(scored)

    ranked.sort(
        key=lambda r: (
            -float(r.get("final_score", 0.0)),
            -float(r.get("tfidf_score", 0.0)),
            int(r.get("order", 0)),
        )
    )
    candidate_records = ranked[: min(len(ranked), max(top_k_scan * 3, top_k_scan))]
    candidates = [_record_text(r) for r in candidate_records]
    cand_scores = [float(r.get("final_score", 0.0)) for r in candidate_records]
    selected = _select_diverse_relevant_chunks(
        candidates,
        scores=cand_scores,
        max_chunks=top_k_scan,
        max_context_chars=max_context_chars,
    )
    return selected if selected else [_record_text(candidate_records[0])]


def _select_compact_context_chunks(
    chunks: Sequence[str],
    max_chunks: int = TOP_K_RETRIEVAL,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> List[str]:
    """
    Keep only compact, high-value chunks:
    - remove empty/duplicate fragments
    - enforce max chunk count and total context budget
    """
    out: List[str] = []
    seen = set()
    total = 0
    for c in chunks:
        if not c:
            continue
        s = str(c).strip()
        if not s or s in seen:
            continue
        if total + len(s) > max_context_chars and out:
            break
        out.append(s)
        seen.add(s)
        total += len(s)
        if len(out) >= max_chunks or total >= max_context_chars:
            break
    return out


def _select_diverse_relevant_chunks(
    chunks: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    max_chunks: int = TOP_K_RETRIEVAL,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    lambda_rel: float = 0.72,
    near_duplicate_threshold: float = 0.93,
) -> List[str]:
    """
    MMR-like selector:
    - keeps relevance to query (scores)
    - discourages redundancy via chunk-to-chunk cosine similarity
    """
    base = _select_compact_context_chunks(chunks, max_chunks=max_chunks * 3, max_context_chars=max_context_chars * 2)
    if not base:
        return []
    if len(base) == 1:
        return base

    rel = np.array(list(scores[: len(base)]) if scores else [1.0] * len(base), dtype=np.float32)
    if rel.size != len(base):
        rel = np.ones(len(base), dtype=np.float32)
    if np.max(rel) > np.min(rel):
        rel = (rel - np.min(rel)) / (np.max(rel) - np.min(rel))
    else:
        rel = np.ones_like(rel)

    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_df=1.0, sublinear_tf=True)
        mat = vec.fit_transform(base)
        pair = cosine_similarity(mat, mat)
    except ValueError:
        pair = np.eye(len(base), dtype=np.float32)

    chosen: List[int] = []
    budget = 0
    remaining = set(range(len(base)))

    while remaining and len(chosen) < max_chunks and budget < max_context_chars:
        if not chosen:
            nxt = max(remaining, key=lambda i: float(rel[i]))
        else:
            def mmr(i: int) -> float:
                max_sim = max(float(pair[i, j]) for j in chosen) if chosen else 0.0
                if max_sim >= near_duplicate_threshold:
                    return -1e9
                return lambda_rel * float(rel[i]) - (1.0 - lambda_rel) * max_sim
            nxt = max(remaining, key=mmr)
            if mmr(nxt) < -1e8:
                break
        txt = base[nxt]
        if budget + len(txt) > max_context_chars and chosen:
            break
        chosen.append(nxt)
        remaining.remove(nxt)
        budget += len(txt)

    return [base[i] for i in chosen] if chosen else _select_compact_context_chunks(base, max_chunks=max_chunks, max_context_chars=max_context_chars)


def _faiss_index_path(doc_id: str) -> str:
    return os.path.join(FAISS_DIR, f"{doc_id}.index")


def _faiss_meta_path(doc_id: str) -> str:
    return os.path.join(FAISS_DIR, f"{doc_id}.json")


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None and SentenceTransformer is not None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _faiss_available() -> bool:
    global faiss, _faiss_import_failed
    if _faiss_import_failed:
        return False
    if faiss is None:
        try:
            import faiss as _faiss
            faiss = _faiss
        except Exception:
            _faiss_import_failed = True
            return False
    return True


def _doc_id_from_text(doc_text: str) -> str:
    payload = (RAG_INDEX_VERSION + "\n" + (doc_text or "")).encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


def _indexable_chunk_records(chunk_records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen = set()
    for record in chunk_records:
        text = _record_text(record)
        if not text or text in seen:
            continue
        if _is_reference_record(record):
            continue
        seen.add(text)
        out.append(
            {
                "text": text,
                "kind": _record_kind(record) or "body",
                "heading": _record_heading(record) or None,
                "order": int(record.get("order", len(out))),
            }
        )
    return out


def _debug_print_indexed_chunks(records: Sequence[Dict[str, object]], limit: int = 10) -> None:
    pass


def _ensure_indexed_in_vector_db(doc_text: str, chunk_records: Sequence[Dict[str, object]]) -> Optional[str]:
    """
    Index chunks once per document hash into FAISS.
    Returns doc_id when vector index is ready, else None.
    """
    if not _faiss_available():
        return None
    embedder = _get_embedding_model()
    if embedder is None:
        return None

    doc_id = _doc_id_from_text(doc_text)
    if doc_id in _indexed_doc_ids:
        return doc_id

    with _index_lock:
        if doc_id in _indexed_doc_ids:
            return doc_id

        os.makedirs(FAISS_DIR, exist_ok=True)
        idx_path = _faiss_index_path(doc_id)
        meta_path = _faiss_meta_path(doc_id)

        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            indexable_records = _indexable_chunk_records(chunk_records)
            chunk_list = [record["text"] for record in indexable_records]
            if not chunk_list:
                return None

            embeddings = embedder.encode(
                chunk_list,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            emb = np.asarray(embeddings, dtype=np.float32)
            if emb.ndim != 2 or emb.shape[0] == 0:
                return None

            index = faiss.IndexFlatIP(int(emb.shape[1]))
            index.add(emb)
            faiss.write_index(index, idx_path)

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"chunks": chunk_list, "records": indexable_records}, f, ensure_ascii=False)

        _indexed_doc_ids.add(doc_id)
        return doc_id


def _retrieve_with_vector_db(
    doc_text: str,
    question: str,
    chunk_records: Sequence[Dict[str, object]],
    top_k_scan: int,
) -> Optional[List[Dict[str, object]]]:
    """Semantic retrieval via FAISS with chunk metadata preserved."""
    if not question.strip():
        return [dict(record) for record in chunk_records[: min(3, len(chunk_records))]]

    doc_id = _ensure_indexed_in_vector_db(doc_text, chunk_records)
    embedder = _get_embedding_model()
    if not doc_id or embedder is None or not _faiss_available():
        return None

    idx_path = _faiss_index_path(doc_id)
    meta_path = _faiss_meta_path(doc_id)
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return None

    try:
        index = faiss.read_index(idx_path)
    except Exception:
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        chunk_list = meta.get("chunks", [])
        record_list = meta.get("records") or [
            {"text": chunk, "kind": "body", "order": i}
            for i, chunk in enumerate(chunk_list)
        ]
    except Exception:
        return None
    if not chunk_list or not record_list:
        return None

    _debug_print_indexed_chunks(record_list)

    q_emb = embedder.encode(
        [question],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    q_arr = np.asarray(q_emb, dtype=np.float32)
    if q_arr.ndim == 1:
        q_arr = q_arr.reshape(1, -1)

    k = min(max(top_k_scan, TOP_K_RETRIEVAL, 1), len(chunk_list))
    try:
        _dist, ids = index.search(q_arr, k)
    except Exception:
        return None

    ids_row = ids[0] if len(ids) > 0 else []
    dist_row = _dist[0] if len(_dist) > 0 else []
    candidates: List[Dict[str, object]] = []
    for pos, i in enumerate(ids_row):
        ii = int(i)
        if not (0 <= ii < len(record_list)):
            continue
        scored = dict(record_list[ii])
        scored["semantic_score"] = float(dist_row[pos]) if pos < len(dist_row) else 0.0
        candidates.append(scored)
    return candidates if candidates else None


def _augment_candidates_for_intent(
    question: str,
    intent: str,
    candidates: Sequence[Dict[str, object]],
    all_records: Sequence[Dict[str, object]],
    tfidf_by_text: Dict[str, float],
) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    seen = set()

    def add_record(record: Dict[str, object]) -> None:
        text = _record_text(record)
        if not text or text in seen or _is_reference_record(record):
            return
        scored = dict(record)
        scored.setdefault("semantic_score", 0.0)
        scored["tfidf_score"] = float(tfidf_by_text.get(text, 0.0))
        merged.append(scored)
        seen.add(text)

    for record in candidates:
        add_record(record)

    if intent == "summary":
        for record in _priority_lead_chunk_records(question, all_records):
            add_record(record)
        for record in all_records:
            if _record_kind(record) == "conclusion":
                add_record(record)
                break
    elif intent == "clause":
        clause_records = [record for record in all_records if _is_clause_like_record(record)]
        clause_records.sort(key=lambda r: -float(tfidf_by_text.get(_record_text(r), 0.0)))
        for record in clause_records[: max(TOP_K_RETRIEVAL + 2, 6)]:
            add_record(record)

    lexical_records = [record for record in all_records if not _is_reference_record(record)]
    lexical_records.sort(key=lambda r: -float(tfidf_by_text.get(_record_text(r), 0.0)))
    for record in lexical_records[: max(TOP_K_RETRIEVAL + 2, 6)]:
        add_record(record)

    return merged


def _retrieve_ranked_chunk_records(
    doc_text: str,
    question: str,
    chunk_records: Sequence[Dict[str, object]],
    top_k_scan: int = TOP_K_RETRIEVAL,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> List[Dict[str, object]]:
    records = [dict(record) for record in chunk_records if _record_text(record)]
    if not records:
        return []

    q = (question or "").strip()
    if not q:
        return _select_compact_context_record_items(records, min(3, len(records)), max_context_chars)

    eligible_records = [record for record in records if not _is_reference_record(record)]
    if not eligible_records:
        eligible_records = records

    intent = _classify_query_intent(q)
    semantic_k = min(len(eligible_records), max(top_k_scan * 4, 12))
    semantic_candidates = _retrieve_with_vector_db(doc_text, q, eligible_records, semantic_k) or []
    semantic_available = bool(semantic_candidates)

    all_texts = [_record_text(record) for record in eligible_records]
    all_tfidf_scores = _compute_tfidf_similarities(q, all_texts)
    tfidf_by_text = {
        all_texts[i]: float(all_tfidf_scores[i])
        for i in range(min(len(all_texts), len(all_tfidf_scores)))
    }

    candidates = _augment_candidates_for_intent(q, intent, semantic_candidates, eligible_records, tfidf_by_text)
    if not candidates:
        candidates = [dict(record) for record in eligible_records]
        for record in candidates:
            record.setdefault("semantic_score", 0.0)
            record["tfidf_score"] = float(tfidf_by_text.get(_record_text(record), 0.0))

    semantic_scores = [float(record.get("semantic_score", 0.0)) for record in candidates]
    tfidf_scores = [float(record.get("tfidf_score", tfidf_by_text.get(_record_text(record), 0.0))) for record in candidates]
    semantic_norm = _normalize_score_array(semantic_scores)
    tfidf_norm = _normalize_score_array(tfidf_scores)

    ranked: List[Dict[str, object]] = []
    semantic_weight = 1.15 if semantic_available else 0.0
    lexical_weight = 0.18 if semantic_available else 0.85

    for idx, record in enumerate(candidates):
        scored = dict(record)
        section_offset = _section_score_offset(record)
        intent_offset = _intent_score_offset(intent, record)
        sem_norm = float(semantic_norm[idx]) if idx < len(semantic_norm) else 0.0
        lex_norm = float(tfidf_norm[idx]) if idx < len(tfidf_norm) else 0.0
        scored["semantic_score"] = float(semantic_scores[idx]) if idx < len(semantic_scores) else 0.0
        scored["tfidf_score"] = float(tfidf_scores[idx]) if idx < len(tfidf_scores) else 0.0
        scored["section_score"] = section_offset
        scored["intent_score"] = intent_offset
        scored["final_score"] = semantic_weight * sem_norm + lexical_weight * lex_norm + section_offset + intent_offset
        ranked.append(scored)

    ranked.sort(
        key=lambda r: (
            -float(r.get("final_score", 0.0)),
            -float(r.get("semantic_score", 0.0)),
            -float(r.get("tfidf_score", 0.0)),
            int(r.get("order", 0)),
        )
    )

    ranked_texts = [_record_text(record) for record in ranked]
    ranked_scores = [float(record.get("final_score", 0.0)) for record in ranked]
    selected_texts = _select_diverse_relevant_chunks(
        ranked_texts,
        scores=ranked_scores,
        max_chunks=max(TOP_K_RETRIEVAL, top_k_scan),
        max_context_chars=max_context_chars,
    )
    selected_set = set(selected_texts)
    selected_records = [record for record in ranked if _record_text(record) in selected_set]
    selected_records.sort(key=lambda r: selected_texts.index(_record_text(r)))
    return selected_records


def _build_messages(context: str, question: str, history: Optional[Sequence[dict]]) -> list:
    system = (
        "You are a document-grounded question answering system.\n\n"
        "Answer ONLY using the provided context.\n\n"
        "Rules:\n\n"
        "* Do not use outside knowledge.\n"
        "* If the context clearly supports an answer, answer directly and confidently.\n"
        "* Only say \"The document does not clearly specify this.\" when the evidence is weak, "
        "ambiguous, contradictory, or missing.\n"
        "* Prefer information from:\n"
        "  Title\n"
        "  Abstract\n"
        "  Introduction\n"
        "  over later sections.\n"
        "* Be concise and factual.\n"
        "* Do not speculate.\n"
        "* Do not infer document purpose from isolated discussion paragraphs.\n"
        "* If asked for the document purpose, prioritize:\n"
        "  abstract,\n"
        "  introduction,\n"
        "  and conclusion sections.\n"
        "* For document-purpose or summary questions, synthesize the supported content into a direct "
        "summary answer instead of leading with a disclaimer.\n"
        "* If title, abstract, or introduction clearly state the answer, use them and answer in 1-3 "
        "clear sentences."
    )
    messages = [{"role": "system", "content": system}]
    user_block = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    messages.append({"role": "user", "content": user_block[:15000]})
    return messages


def _extract_potential_fact_tokens(text: str) -> set:
    """
    Extract high-risk hallucination tokens:
    - amounts/currency/percent
    - dates
    - durations/years/months
    - standalone long numbers
    """
    s = text or ""
    tokens = set()

    patterns = [
        r"(?:Rs\.?|INR|₹|USD|\$)\s*[\d,]+(?:\.\d+)?",
        r"[\d,]+(?:\.\d+)?\s*(?:%|percent)",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:\d{4})\b",
        r"\b\d+\s*(?:years?|months?|days?)\b",
        r"\b\d{3,}\b",
    ]
    for p in patterns:
        for m in re.findall(p, s, flags=re.IGNORECASE):
            tok = re.sub(r"\s+", " ", str(m).strip())
            if tok:
                tokens.add(tok.lower())
    return tokens


def _find_unsupported_tokens(answer: str, context: str) -> List[str]:
    ans_tokens = _extract_potential_fact_tokens(answer)
    ctx_tokens = _extract_potential_fact_tokens(context)
    unsupported = [t for t in ans_tokens if t not in ctx_tokens]
    return unsupported[:8]


def _generate_answer_with_guardrails(messages: list) -> str:
    resp = ollama_chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": 0.1, "num_predict": 320},
    )
    out = (getattr(resp.message, "content", None) or "").strip()
    return out if out else "(The model returned an empty answer.)"


def _context_strongly_supports_answer(question: str, context: str) -> bool:
    q = (question or "").strip().lower()
    ctx = context or ""
    if not q or not ctx.strip():
        return False

    if _is_document_summary_question(q):
        support_hits = sum(1 for marker in ("Title:", "Abstract:", "Introduction:") if marker in ctx)
        if support_hits >= 2:
            return True
        if "Abstract:" in ctx and len(ctx) >= 280:
            return True
        if "Introduction:" in ctx and len(ctx) >= 420:
            return True
        return False

    if any(marker in ctx for marker in ("Abstract:", "Introduction:", "Title:")) and len(ctx) >= 350:
        return True
    return False


def _answer_is_overly_defensive(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return False
    return (
        a.startswith("the document does not clearly specify this")
        or a.startswith("not clearly specified")
        or a.startswith("it is not clearly specified")
    )


def _repair_answer_if_needed(question: str, answer: str, context: str) -> str:
    # Fast path: skip repair unless answer includes risky factual tokens
    # (numbers/dates/currency/durations) that can hallucinate.
    if not _extract_potential_fact_tokens(answer):
        return answer
    unsupported = _find_unsupported_tokens(answer, context)
    if not unsupported:
        return answer

    repair_messages = [
        {
            "role": "system",
            "content": (
                "Revise the answer so every factual token is grounded in DOCUMENT EXCERPTS. "
                "Do not add new facts. If information is absent, explicitly say "
                "'The document does not clearly specify this.'"
            ),
        },
        {
            "role": "user",
            "content": (
                f"DOCUMENT EXCERPTS:\n{context[:12000]}\n\n"
                f"QUESTION:\n{question}\n\n"
                f"DRAFT ANSWER:\n{answer}\n\n"
                f"UNSUPPORTED TOKENS FOUND:\n{', '.join(unsupported)}\n\n"
                "Return corrected final answer only."
            ),
        },
    ]
    try:
        repaired = _generate_answer_with_guardrails(repair_messages)
        # If still unsupported, prefer conservative original with explicit caution.
        if _find_unsupported_tokens(repaired, context):
            return (
                f"{answer}\n\nNote: Some numeric/date details may be absent in the provided excerpts. "
                "Treat unspecified values as not clearly specified by the document."
            )
        return repaired
    except Exception:
        return answer


def _revise_overly_defensive_answer_if_needed(question: str, answer: str, context: str) -> str:
    if not _answer_is_overly_defensive(answer):
        return answer
    if not _context_strongly_supports_answer(question, context):
        return answer

    revise_messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the answer using only the provided context. "
                "If the context clearly supports the answer, respond directly and confidently. "
                "Do not lead with uncertainty when title, abstract, or introduction provide clear evidence. "
                "Do not add any facts not present in the context. "
                "Use 'The document does not clearly specify this.' only if the context is actually insufficient."
            ),
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context[:12000]}\n\n"
                f"QUESTION:\n{question}\n\n"
                f"CURRENT ANSWER:\n{answer}\n\n"
                "Return a cleaner final answer only."
            ),
        },
    ]
    try:
        revised = _generate_answer_with_guardrails(revise_messages)
        return revised or answer
    except Exception:
        return answer


_FUSED_WORD_LEXICON = {
    "a", "access", "account", "actions", "after", "against", "agreement", "all", "allowance", "and", "any",
    "apart", "are", "arisen", "as", "authority", "be", "because", "before", "between", "by", "called", "can",
    "case", "chaste", "child", "children", "claim", "claims", "clause", "conjugal", "consent", "contained",
    "contract", "could", "custody", "date", "days", "death", "demand", "demands", "differences", "discharge",
    "disputes", "document", "duplicate", "during", "each", "either", "entitled", "except", "exception",
    "executed", "expressly", "facts", "for", "from", "further", "gave", "giving", "guardian", "guardianship",
    "has", "have", "having", "he", "her", "here", "hereafter", "herein", "hers", "him", "his", "husband",
    "if", "in", "incurred", "indemnified", "indemnifies", "intend", "is", "it", "its", "keep", "keeps",
    "liable", "liabilities", "liability", "life", "lifetime", "limitations", "live", "living",
    "made", "maintenance", "majority", "marriage", "may", "months", "namely", "no", "not", "notice",
    "notwithstanding", "now", "obligation", "of", "on", "or", "other", "otherwise", "out", "over", "part",
    "parties", "party", "pay", "payable", "payment", "present", "presents", "prior", "proceeding", "provided",
    "question", "reconciliation", "relevant", "resident", "respectively", "responsibilities", "responsibility",
    "restitution", "return", "revoked", "right", "rights", "same", "separate", "separately", "settlement",
    "shall", "she", "society", "specified", "state", "stated", "states", "stop", "such", "support", "that",
    "the", "their", "them", "there", "thereafter", "therein", "thereis", "thereof", "this", "times", "to",
    "together", "under", "unless", "use", "void", "want", "wants", "was", "were", "whereas", "wife", "will",
    "witnesses", "witnesseth", "with", "without", "years",
}


def _split_fused_token(token: str) -> str:
    raw = token or ""
    core = re.sub(r"[^A-Za-z]", "", raw)
    if len(core) < 14:
        return token

    s = core.lower()
    n = len(s)
    best: List[Optional[tuple]] = [None] * (n + 1)
    best[0] = (0, 0, [])
    max_word_len = max(len(w) for w in _FUSED_WORD_LEXICON)

    for i in range(n):
        if best[i] is None:
            continue
        score_i, covered_i, parts_i = best[i]
        upper = min(n, i + max_word_len)
        for j in range(i + 1, upper + 1):
            piece = s[i:j]
            is_word = piece in _FUSED_WORD_LEXICON
            score_j = score_i + (8 if is_word else -3 * len(piece))
            covered_j = covered_i + (len(piece) if is_word else 0)
            cand = (score_j, covered_j, parts_i + [piece])
            cur = best[j]
            if cur is None or cand[0] > cur[0] or (cand[0] == cur[0] and cand[1] > cur[1]):
                best[j] = cand

    if best[n] is None:
        return token
    _, covered, parts = best[n]
    if covered / max(1, n) < 0.75 or len(parts) < 3:
        return token
    # Guardrail: reject noisy segmentations like "m a in t a in".
    if any(len(p) == 1 and p not in {"a", "i"} for p in parts):
        return token
    short_ratio = sum(1 for p in parts if len(p) <= 2) / max(1, len(parts))
    if short_ratio > 0.34:
        return token
    if len(parts) > max(8, n // 3):
        return token

    spaced = " ".join(parts)
    if raw and raw[0].isupper():
        spaced = spaced[:1].upper() + spaced[1:]
    return spaced


def _restore_spacing_artifacts(answer: str) -> str:
    """
    Fix common OCR/extraction artifacts where many words are fused together.
    Only reformats spacing; does not add facts.
    """
    if not answer:
        return answer

    pattern = re.compile(r"\b[A-Za-z]{14,}\b")
    return pattern.sub(lambda m: _split_fused_token(m.group(0)), answer)


def rag_answer(
    document_text: str,
    question: str,
    history: Optional[Sequence[dict]] = None,
) -> str:
    """
    Run retrieval over document_text, then answer with the local Ollama model.

    history: optional list of {"role": "user"|"assistant", "content": "..."} for short multi-turn.
    """
    doc = _normalize_whitespace(document_text or "")
    doc = _fix_broken_text(doc)
    if len(doc) < 25:
        return "The document is empty or too short to search. Upload a PDF with extractable text."

    q = (question or "").strip()
    if not q:
        return "Please enter a question."

    doc = doc[:MAX_DOC_CHARS]
    chunk_records = _build_chunk_records(doc)
    if not chunk_records:
        return "Could not process the document text."

    priority_records = _priority_lead_chunk_records(q, chunk_records)
    ranked_records = _retrieve_ranked_chunk_records(
        doc,
        q,
        chunk_records,
        top_k_scan=TOP_K_RETRIEVAL,
        max_context_chars=MAX_CONTEXT_CHARS,
    )
    ranked_by_text = {_record_text(record): record for record in ranked_records}
    merged_records = [ranked_by_text.get(_record_text(record), record) for record in priority_records]
    priority_texts = {_record_text(record) for record in priority_records}
    for record in ranked_records:
        if _record_text(record) not in priority_texts:
            merged_records.append(record)

    final_records = _select_compact_context_record_items(
        merged_records,
        max_chunks=TOP_K_RETRIEVAL + len(priority_records),
        max_context_chars=MAX_CONTEXT_CHARS,
    )
    relevant = [_record_text(record) for record in final_records]
    print(f"[RETRIEVE] question: {q!r}")
    print(f"[RETRIEVE] chunks_selected: {len(final_records)}")
    for i, record in enumerate(final_records, 1):
        preview = _record_text(record)[:80].replace("\n", " ")
        print(f"[RETRIEVE]   rank {i}: preview={preview!r}")

    context = "\n\n---\n\n".join(relevant)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    messages = _build_messages(context, q, history)
    try:
        out = _generate_answer_with_guardrails(messages)
        print(f"[GENERATE] raw_answer: {out[:200]!r}")

        out_before_repair = out
        out = _repair_answer_if_needed(q, out, context)
        if out != out_before_repair:
            print(f"[GUARDRAIL] repair triggered")
            print(f"[GUARDRAIL] before: {out_before_repair[:150]!r}")
            print(f"[GUARDRAIL] after:  {out[:150]!r}")
        else:
            print("[GUARDRAIL] no unsupported claims found, no repair triggered")

        out = _revise_overly_defensive_answer_if_needed(q, out, context)
        out = _restore_spacing_artifacts(out)
        return out
    except Exception as e:
        return f"Could not reach Ollama model ({OLLAMA_MODEL}). Is it running locally? Details: {e}"


__all__ = [
    "OLLAMA_MODEL",
    "rag_answer",
    "chunk_document",
    "retrieve_relevant_chunks",
]
