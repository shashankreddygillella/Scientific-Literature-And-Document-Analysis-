#!/usr/bin/env python
# coding: utf-8
"""
Flask Web Application for Scientific Document Summarization
"""

import os
import json
import base64
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, pipeline
)
import nltk
from nltk.tokenize import sent_tokenize
import networkx as nx
from werkzeug.utils import secure_filename
import PyPDF2
from io import BytesIO
import re
import unicodedata
from collections import defaultdict
from collections import defaultdict
from ollama import chat as ollama_chat
import sqlite3

from scientific_rag import rag_answer
import re
# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'lexsumm-dev-secret-change-me')
ALLOWED_EXTENSIONS = {'pdf'}

# Model configuration
# Lightweight MiniLM encoder for faster sentence embeddings.
DOMAIN_ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
SUMMARIZER = 'google/pegasus-xsum'
MAX_INPUT_TOKENS = 512
MAX_TARGET_TOKENS = 128

# Global variables for models
device = torch.device("cpu")
model_finetuned = None
tokenizer_finetuned = None
summarizer_pipe_finetuned = None
model_pegasus = None
tokenizer_pegasus = None
summarizer_pipe_pegasus = None
minilm_tokenizer = None
minilm_encoder = None


def _db_connect():
    return sqlite3.connect('signup.db')


def init_chat_history_db():
    """Create chat history tables if they do not exist."""
    con = _db_connect()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(conversation_id) REFERENCES chat_conversations(id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_conv_user ON chat_conversations(username)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_msg_conv ON chat_messages(conversation_id)")
    con.commit()
    con.close()


def _current_username():
    user = session.get('username')
    return str(user).strip() if user else None


def _create_conversation(username: str, first_question: str = '') -> int:
    title = (first_question or '').strip()
    if not title:
        title = 'New conversation'
    if len(title) > 70:
        title = title[:67].rstrip() + '...'
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO chat_conversations (username, title) VALUES (?, ?)",
        (username, title)
    )
    conv_id = int(cur.lastrowid)
    con.commit()
    con.close()
    return conv_id


def _touch_conversation(con, conversation_id: int):
    cur = con.cursor()
    cur.execute(
        "UPDATE chat_conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (conversation_id,)
    )


def _save_message(conversation_id: int, role: str, content: str):
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO chat_messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content)
    )
    _touch_conversation(con, conversation_id)
    con.commit()
    con.close()


init_chat_history_db()

# Utility functions
def chunk_text(text: str, max_tokens: int, tokenizer) -> list:
    """Simple sentence-based chunker."""
    sents = sent_tokenize(text)
    chunks, cur = [], []
    cur_len = 0
    
    for s in sents:
        l = len(tokenizer.encode(s, add_special_tokens=False))
        if cur_len + l > max_tokens and cur:
            chunks.append(' '.join(cur))
            cur, cur_len = [s], l
        else:
            cur.append(s)
            cur_len += l
    
    if cur:
        chunks.append(' '.join(cur))
    
    return chunks

def chunk_text_for_ollama(text, max_chunk_size=1000):
    """Chunking for Ollama based on character length"""
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def _clean_ollama_summary_output(text: str) -> str:
    """Normalize Ollama output to summary-only plain text."""
    if not text or not isinstance(text, str):
        return ""

    cleaned = text.strip()
    # Remove common markdown emphasis markers that leak from chat-style answers.
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)

    # Drop common meta sections that are not part of the actual summary.
    cut_patterns = [
        r"(?im)^\s*to further refine.*$",
        r"(?im)^\s*could you tell me.*$",
        r"(?im)^\s*what is the specific.*$",
        r"(?im)^\s*are there any particular.*$",
        r"(?im)^\s*key elements\s*&\s*considerations\s*:.*$",
    ]
    for pattern in cut_patterns:
        match = re.search(pattern, cleaned)
        if match:
            cleaned = cleaned[:match.start()].strip()

    # If model emits labeled prefixes, keep only the body.
    cleaned = re.sub(r"(?is)^\s*(final\s+summary\s*:)\s*", "", cleaned).strip()

    return cleaned.strip()

def parse_top_k_option(raw_value) -> tuple:
    """Parse key-sentence option supporting both int and range strings like '6-8'."""
    default_top_k = 8
    default_range = (6, 8)

    if raw_value is None:
        return default_top_k, default_range

    if isinstance(raw_value, int):
        top_k = max(3, min(raw_value, 20))
        lo = max(3, top_k - 2)
        hi = top_k
        return top_k, (lo, hi)

    s = str(raw_value).strip()
    m = re.fullmatch(r'(\d+)\s*-\s*(\d+)', s)
    if m:
        lo = int(m.group(1))
        hi = int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        lo = max(3, min(lo, 20))
        hi = max(lo, min(hi, 20))
        return hi, (lo, hi)

    try:
        top_k = int(s)
        top_k = max(3, min(top_k, 20))
        lo = max(3, top_k - 2)
        hi = top_k
        return top_k, (lo, hi)
    except Exception:
        return default_top_k, default_range


def detect_input_language(text: str) -> tuple:
    """Heuristic language detection for major scripts used in this project."""
    if not text or not isinstance(text, str):
        return 'unknown', 'Unknown'

    sample = text[:6000]
    latin = len(re.findall(r'[A-Za-z]', sample))
    tamil = len(re.findall(r'[\u0B80-\u0BFF]', sample))
    telugu = len(re.findall(r'[\u0C00-\u0C7F]', sample))
    devanagari = len(re.findall(r'[\u0900-\u097F]', sample))

    counts = {
        'english': latin,
        'tamil': tamil,
        'telugu': telugu,
        'hindi': devanagari,
    }
    best_lang = max(counts, key=counts.get)
    best_count = counts[best_lang]
    total = sum(counts.values())
    if total == 0 or best_count == 0:
        return 'unknown', 'Unknown'

    # Ensure dominant script is significant.
    if best_count / total < 0.45:
        return 'unknown', 'Unknown'

    names = {
        'english': 'English',
        'tamil': 'Tamil',
        'telugu': 'Telugu',
        'hindi': 'Hindi',
    }
    return best_lang, names[best_lang]


def translate_summary_to_english(summary_text: str, source_language_name: str) -> str:
    """Translate summary to English using Ollama chat if available."""
    if not summary_text or not isinstance(summary_text, str):
        return ''

    prompt = f"""
Translate the following summary from {source_language_name} to English.

Rules:
- Keep all scientific facts unchanged.
- Keep numbers, names, dates, and amounts exact.
- Return plain text only.
- Do not add headings, bullets, notes, or questions.

TEXT:
{summary_text}

ENGLISH TRANSLATION:
"""
    try:
        response = ollama_chat(
            model='gemma3:1b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}
        )
        return _clean_ollama_summary_output(response.message.content)
    except Exception as e:
        print(f"[WARNING] English translation failed: {e}")
        return ''


def _language_script_ratio(text: str, lang_code: str) -> float:
    """Estimate how much text matches the expected script for a language."""
    if not text:
        return 0.0
    patterns = {
        'english': r'[A-Za-z]',
        'tamil': r'[\u0B80-\u0BFF]',
        'telugu': r'[\u0C00-\u0C7F]',
        'hindi': r'[\u0900-\u097F]',
    }
    pattern = patterns.get(lang_code)
    if not pattern:
        return 0.0
    script_chars = len(re.findall(pattern, text))
    alpha_chars = len(re.findall(r'[A-Za-z\u0B80-\u0BFF\u0C00-\u0C7F\u0900-\u097F]', text))
    if alpha_chars == 0:
        return 0.0
    return script_chars / alpha_chars


def translate_summary_to_language(summary_text: str, target_language_name: str) -> str:
    """Translate summary to a target language using Ollama."""
    if not summary_text or not isinstance(summary_text, str):
        return ''

    prompt = f"""
Translate the following scientific summary to {target_language_name}.

Rules:
- Keep all scientific facts unchanged.
- Keep numbers, names, dates, and amounts exact.
- Return plain text only.
- Do not add headings, bullets, notes, or questions.

TEXT:
{summary_text}

TRANSLATION:
"""
    try:
        response = ollama_chat(
            model='gemma3:1b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}
        )
        return _clean_ollama_summary_output(response.message.content)
    except Exception as e:
        print(f"[WARNING] Translation to {target_language_name} failed: {e}")
        return ''


def build_bilingual_summaries(
    summary: str,
    summary_format: str,
    input_language_code: str,
    input_language_name: str
) -> tuple:
    """
    Build source-language + English summaries for non-English input.
    Returns: (source_summary, source_display, english_summary, english_display, combined_display, bilingual)
    """
    source_summary = summary
    source_summary_display = format_summary_display(source_summary, summary_format)
    english_summary = ''
    english_summary_display = ''
    bilingual = input_language_code not in ('english', 'unknown')

    if not bilingual:
        return (
            source_summary,
            source_summary_display,
            english_summary,
            english_summary_display,
            source_summary_display,
            False,
        )

    # If the generated summary is not really in the expected language, force translation.
    source_ratio = _language_script_ratio(source_summary, input_language_code)
    if source_ratio < 0.35:
        translated_source = translate_summary_to_language(source_summary, input_language_name)
        if translated_source:
            source_summary = translated_source
            source_summary_display = format_summary_display(source_summary, summary_format)

    english_summary = translate_summary_to_english(source_summary, input_language_name)
    if not english_summary:
        english_summary = summary
    english_summary_display = format_summary_display(english_summary, summary_format)

    combined_display = (
        f"{input_language_name} summary:\n{source_summary_display}\n\n"
        f"English summary:\n{english_summary_display}"
    )
    return (
        source_summary,
        source_summary_display,
        english_summary,
        english_summary_display,
        combined_display,
        True,
    )


def summarize_chunk_ollama(chunk, target_range: tuple = (6, 8)):
    low, high = target_range
    prompt = f"""
You are a scientific-document summarization engine.
Task: produce only a concise factual summary of the input.

Rules:
- Return plain text only.
- Do not include headings, labels, markdown, bullet points, or numbering.
- Do not add commentary, suggestions, or follow-up questions.
- Do not mention missing information.
- Preserve key scientific facts: methods, datasets, metrics, findings, and conditions.
- Keep it short (about {max(2, low - 1)}-{max(3, high - 2)} sentences).

TEXT:
{chunk}

SUMMARY:
"""
    try:
        response = ollama_chat(
            model='gemma3:1b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}
        )
        return _clean_ollama_summary_output(response.message.content)
    except Exception as e:
        print(f"Ollama chunk summarization error: {e}")
        return ""

def final_summary_ollama(all_summaries):
    return final_summary_ollama_with_range(all_summaries, (6, 8))


def final_summary_ollama_with_range(all_summaries, target_range: tuple = (6, 8)):
    combined = " ".join(all_summaries)
    low, high = target_range

    prompt = f"""
You are a scientific-document summarization engine.
Combine the partial summaries into one final summary.

Rules:
- Return plain text only.
- Output exactly one paragraph.
- Do not include headings, labels, markdown, bullet points, or numbering.
- Do not ask questions.
- Do not add advice or extra explanation.
- Keep only factual scientific content from the input.
- Output between {low} and {high} sentences.

SUMMARIES:
{combined}

FINAL SUMMARY:
"""
    try:
        response = ollama_chat(
            model='gemma3:1b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}
        )
        return _clean_ollama_summary_output(response.message.content)
    except Exception as e:
        print(f"Ollama final summarization error: {e}")
        return combined


def textrank_extract(text: str, top_n: int = 8) -> list:
    """TextRank implementation for extractive summarization."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        # Fallback to simple sentence selection
        sents = sent_tokenize(text)
        return sents[:top_n] if len(sents) > top_n else sents
    
    sents = sent_tokenize(text)
    if len(sents) <= top_n:
        return sents
    
    # Embed sentences
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(sents, show_progress_bar=False)
    
    # Compute similarity matrix
    sim = np.dot(embeddings, embeddings.T)
    
    # Build graph and compute PageRank
    G = nx.from_numpy_array(sim)
    try:
        scores = nx.pagerank_numpy(G)
    except:
        scores = nx.pagerank(G)
    
    ranked = sorted(((scores[i], s) for i, s in enumerate(sents)), reverse=True)
    return [s for _, s in ranked[:top_n]]


def sentence_embedding(texts: list, tokenizer, encoder) -> np.ndarray:
    """Generate sentence embeddings using a transformer model."""
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    enc = {k: v.to(encoder.device) for k, v in enc.items()}
    
    with torch.no_grad():
        out = encoder(**enc)
    
    # Mean pooling
    attn_mask = enc['attention_mask'].unsqueeze(-1)
    reps = (out.last_hidden_state * attn_mask).sum(1) / attn_mask.sum(1)
    
    return reps.cpu().numpy()


def textrank_with_minilm(text: str, top_n: int = 12) -> list:
    """TextRank using MiniLM embeddings."""
    if minilm_tokenizer is None or minilm_encoder is None:
        return textrank_extract(text, top_n)
    
    sents = sent_tokenize(text)
    if len(sents) <= top_n:
        return sents
    
    # Get embeddings using MiniLM
    embeds = sentence_embedding(sents, minilm_tokenizer, minilm_encoder)
    
    # Compute similarity and PageRank
    sim = np.dot(embeds, embeds.T)
    G = nx.from_numpy_array(sim)
    try:
        scores = nx.pagerank_numpy(G)
    except:
        scores = nx.pagerank(G)
    
    ranked = sorted(((scores[i], s) for i, s in enumerate(sents)), reverse=True)
    return [s for _, s in ranked[:top_n]]


def compute_sentence_importance_lime(text: str, summary: str, top_k_sent: int = 12, use_minilm: bool = True) -> dict:
    """Compute LIME-like importance scores for sentences using perturbation."""
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return {}
    
    # Get which sentences were extracted
    if use_minilm and minilm_tokenizer and minilm_encoder:
        extracted = textrank_with_minilm(text, top_n=top_k_sent)
    else:
        extracted = textrank_extract(text, top_n=top_k_sent)
    
    importance_scores = {}
    
    # Simple approach: Use TextRank scores and extraction status
    for i, sent in enumerate(sentences):
        # Base importance from whether sentence was extracted
        if sent in extracted:
            base_score = 0.8
        else:
            base_score = 0.3
        
        # Boost if sentence contains key terms (amounts, dates, domain terms)
        if re.search(r'రూ\.|₹|Rs\.|[\d,]+', sent):
            base_score += 0.1
        
        # Normalize
        importance_scores[i] = min(base_score, 1.0)
    
    return importance_scores


def compute_shap_values_simple(text: str, summary: str, top_k_sent: int = 12, use_minilm: bool = True) -> dict:
    """Compute SHAP-like values using sentence position and extraction status."""
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return {}
    
    # Get extracted sentences
    if use_minilm and minilm_tokenizer and minilm_encoder:
        extracted = textrank_with_minilm(text, top_n=top_k_sent)
    else:
        extracted = textrank_extract(text, top_n=top_k_sent)
    
    shap_values = {}
    total_sentences = len(sentences)
    
    for i, sent in enumerate(sentences):
        # Base SHAP value from extraction status
        if sent in extracted:
            base_value = 0.7
        else:
            base_value = 0.2
        
        # Position-based contribution (earlier sentences often more important)
        position_factor = 1.0 - (i / max(total_sentences, 1)) * 0.3
        base_value *= position_factor
        
        # Length factor (longer sentences might be more informative)
        length_factor = min(len(sent.split()) / 30.0, 1.0) * 0.2
        base_value += length_factor
        
        shap_values[i] = min(base_value, 1.0)
    
    return shap_values


def preserve_amounts_in_summary(original_text: str, summary: str) -> str:
    """Post-process summary to ensure amounts are preserved."""
    # Extract all amounts with context from original text
    amount_patterns = [
        (r'రూ\.\s*([\d,]+)', 'రూ. {}'),  # Telugu: రూ. 20,00,000
        (r'₹\s*([\d,]+)', '₹ {}'),  # Hindi: ₹ 20,00,000
        (r'Rs\.\s*([\d,]+)', 'Rs. {}'),  # English: Rs. 20,00,000
    ]
    
    amounts_found = []
    for pattern, template in amount_patterns:
        matches = re.finditer(pattern, original_text)
        for match in matches:
            amount = match.group(1)
            amounts_found.append((amount, template.format(amount)))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_amounts = []
    for amount, formatted in amounts_found:
        if amount not in seen:
            seen.add(amount)
            unique_amounts.append((amount, formatted))
    
    # If summary mentions currency but amounts are missing or incomplete
    if 'రూ.' in summary or '₹' in summary or 'Rs.' in summary:
        # Check what amounts are in the summary
        summary_amount_matches = re.findall(r'(రూ\.|₹|Rs\.)\s*([\d,]+|)', summary)
        
        # Find positions where currency is mentioned but amount is missing
        replacements = []
        used_amount_indices = set()
        
        for match in re.finditer(r'(రూ\.|₹|Rs\.)\s*([\d,]*)\s*', summary):
            currency = match.group(1)
            amount_after = match.group(2) if match.group(2) else ''
            
            # If currency symbol exists but amount is missing or very short
            if not amount_after or len(amount_after.replace(',', '')) < 3:
                # Try to find a matching amount from original text
                for idx, (amount, formatted) in enumerate(unique_amounts):
                    if idx in used_amount_indices:
                        continue
                    
                    # Match currency symbol
                    if currency == 'రూ.' and 'రూ.' in formatted:
                        replacements.append((match.start(), match.end(), formatted))
                        used_amount_indices.add(idx)
                        break
                    elif currency == '₹' and '₹' in formatted:
                        replacements.append((match.start(), match.end(), formatted))
                        used_amount_indices.add(idx)
                        break
                    elif currency == 'Rs.' and 'Rs.' in formatted:
                        replacements.append((match.start(), match.end(), formatted))
                        used_amount_indices.add(idx)
                        break
        
        # Apply replacements in reverse order to maintain positions
        for start, end, replacement in reversed(replacements):
            summary = summary[:start] + replacement + summary[end:]
    
    return summary


def preserve_amounts_in_sentences(original_text: str, extracted_sentences: list) -> list:
    """Post-process extracted sentences to ensure amounts are preserved."""
    if not extracted_sentences or not original_text:
        return extracted_sentences
    
    # Extract all amounts with context from original text
    amount_patterns = [
        (r'రూ\.\s*([\d,]+)', 'రూ. {}'),  # Telugu: రూ. 20,00,000
        (r'₹\s*([\d,]+)', '₹ {}'),  # Hindi: ₹ 20,00,000
        (r'Rs\.\s*([\d,]+)', 'Rs. {}'),  # English: Rs. 20,00,000
    ]
    
    # Find all amounts in original text with their context
    amounts_map = {}
    for pattern, template in amount_patterns:
        matches = re.finditer(pattern, original_text)
        for match in matches:
            amount = match.group(1) if match.groups() else match.group(0)
            context_start = max(0, match.start() - 50)
            context_end = min(len(original_text), match.end() + 50)
            context = original_text[context_start:context_end]
            amounts_map[amount] = {
                'formatted': template.format(amount) if '{' in template else match.group(0),
                'context': context,
                'position': match.start()
            }
    
    # Enhance each extracted sentence
    enhanced_sentences = []
    for sent in extracted_sentences:
        enhanced_sent = sent
        
        # Check if sentence mentions currency but amount is missing or incomplete
        if 'రూ.' in sent or '₹' in sent or 'Rs.' in sent:
            # Find currency mentions in the sentence
            currency_matches = list(re.finditer(r'(రూ\.|₹|Rs\.)\s*([\d,]*)\s*', sent))
            
            for match in currency_matches:
                currency = match.group(1)
                amount_after = match.group(2) if match.groups() and len(match.groups()) > 1 else ''
                
                # If currency exists but amount is missing or very short
                if not amount_after or len(amount_after.replace(',', '')) < 3:
                    # Try to find matching amount from original text
                    # Look for amounts near this sentence in the original text
                    sent_pos = original_text.find(sent)
                    if sent_pos != -1:
                        # Search in a window around the sentence
                        search_start = max(0, sent_pos - 200)
                        search_end = min(len(original_text), sent_pos + len(sent) + 200)
                        search_window = original_text[search_start:search_end]
                        
                        # Find amounts in this window
                        for amount_key, amount_info in amounts_map.items():
                            if amount_info['position'] >= search_start and amount_info['position'] <= search_end:
                                # Check if this amount matches the currency
                                formatted_amount = amount_info['formatted']
                                if currency in formatted_amount:
                                    # Replace the incomplete amount with the full amount
                                    incomplete_pattern = re.escape(currency) + r'\s*' + (re.escape(amount_after) if amount_after else r'[\d,]*')
                                    if re.search(incomplete_pattern, enhanced_sent):
                                        enhanced_sent = re.sub(
                                            incomplete_pattern,
                                            formatted_amount,
                                            enhanced_sent,
                                            count=1
                                        )
                                        break
        
        # Also check if sentence mentions installment numbers but no amounts
        installment_patterns = [
            (r'మొదటి\s+వాయిదా[:\s]*', 'మొదటి వాయిదా: '),
            (r'రెండవ\s+వాయిదా[:\s]*', 'రెండవ వాయిదా: '),
            (r'మూడవ\s+వాయిదా[:\s]*', 'మూడవ వాయిదా: '),
            (r'నాలుగవ\s+వాయిదా[:\s]*', 'నాలుగవ వాయిదా: '),
        ]
        
        for pattern, label in installment_patterns:
            if re.search(pattern, enhanced_sent):
                # Check if amount follows
                after_match = enhanced_sent[re.search(pattern, enhanced_sent).end():]
                if not re.search(r'రూ\.\s*[\d,]', after_match):
                    # Find the corresponding amount in original text
                    for amount_key, amount_info in amounts_map.items():
                        if label.split(':')[0].strip() in amount_info['context']:
                            # Add the amount to the sentence
                            enhanced_sent = re.sub(
                                pattern,
                                label + amount_info['formatted'] + ' - ',
                                enhanced_sent,
                                count=1
                            )
                            break
        
        enhanced_sentences.append(enhanced_sent)
    
    return enhanced_sentences


# Summary presentation (API + UI)
ALLOWED_SUMMARY_FORMATS = frozenset({
    'paragraphs', 'single_paragraph', 'sentences', 'bullets', 'numbered',
})

# Short English steering phrases prepended to chunked extractive input for Pegasus.
THEME_FOCUS_INSTRUCTIONS = {
    'general': '',
    'facts_timeline': (
        'Summarize with emphasis on chronology, dates, sequences of events, and the factual timeline.'
    ),
    'parties_obligations': (
        'Summarize with emphasis on parties, their roles, rights, duties, and obligations.'
    ),
    'risks_outcomes': (
        'Summarize with emphasis on disputes, risks, remedies, judgments, and outcomes.'
    ),
    'financial_amounts': (
        'Summarize with emphasis on monetary amounts, payments, penalties, and financial terms.'
    ),
}

ALLOWED_THEME_FOCUS = frozenset(THEME_FOCUS_INSTRUCTIONS.keys())


def normalize_summary_options(summary_format: str, theme_focus: str) -> tuple:
    """Validate and return canonical summary_format and theme_focus."""
    fmt = (summary_format or 'paragraphs').strip().lower()
    if fmt not in ALLOWED_SUMMARY_FORMATS:
        fmt = 'paragraphs'
    focus = (theme_focus or 'general').strip().lower()
    if focus not in ALLOWED_THEME_FOCUS:
        focus = 'general'
    return fmt, focus


def format_summary_display(summary: str, summary_format: str) -> str:
    """Shape model summary text for UI (plain text; use white-space: pre-wrap in CSS)."""
    if not summary or not isinstance(summary, str):
        return summary or ''
    s = summary.strip()
    if not s:
        return s
    fmt = (summary_format or 'paragraphs').strip().lower()
    if fmt not in ALLOWED_SUMMARY_FORMATS:
        fmt = 'paragraphs'
    try:
        sentences = [x.strip() for x in sent_tokenize(s) if x.strip()]
    except Exception:
        sentences = [s]
    if not sentences:
        return s
    if fmt == 'single_paragraph':
        return ' '.join(sentences)
    if fmt == 'sentences':
        return '\n'.join(sentences)
    if fmt == 'bullets':
        return '\n'.join(f'• {sent}' for sent in sentences)
    if fmt == 'numbered':
        return '\n'.join(f'{i + 1}. {sent}' for i, sent in enumerate(sentences))
    # paragraphs: group ~2 sentences per block
    chunks = []
    buf = []
    for sent in sentences:
        buf.append(sent)
        if len(buf) >= 2:
            chunks.append(' '.join(buf))
            buf = []
    if buf:
        chunks.append(' '.join(buf))
    return '\n\n'.join(chunks)


def _pick_summarizer_backend(use_pegasus: bool, use_ollama: bool = True):
    """Choose seq2seq model: Pegasus when requested, else fine-tuned if present, else Pegasus."""
    if use_ollama:
        return None, None, None, 'ollama'
    if model_pegasus is None or tokenizer_pegasus is None:
        return None, None, None, 'none'
    if use_pegasus:
        return model_pegasus, tokenizer_pegasus, summarizer_pipe_pegasus, 'pegasus'
    if model_finetuned is not None and tokenizer_finetuned is not None:
        return model_finetuned, tokenizer_finetuned, summarizer_pipe_finetuned, 'finetuned'
    return model_pegasus, tokenizer_pegasus, summarizer_pipe_pegasus, 'pegasus'


def summarize_scientific(
    text: str,
    top_k_sent: int = 12,
    use_minilm: bool = True,
    theme_focus: str = 'general',
    use_pegasus: bool = False,
    use_ollama: bool = True,
    sentence_range: tuple = (6, 8),
) -> str:
    """Summarize a scientific document."""
    try:
        active_model, active_tokenizer, active_pipe, _backend_name = _pick_summarizer_backend(use_pegasus, use_ollama)
        
        # OLLAMA PATH
        if use_ollama:
            print("[DEBUG] Using Ollama (gemma3:1b) for summarization...")
            # For ollama, the user prompt processes the full extracted text chunk by chunk
            chunks = chunk_text_for_ollama(text)
            print(f"Total chunks for Ollama: {len(chunks)}")
            summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                summary = summarize_chunk_ollama(chunk, target_range=sentence_range)
                if summary:
                    summaries.append(summary)
            if not summaries:
                raise Exception("Ollama returned empty summaries for all chunks.")
            print("Creating final summary with Ollama...")
            final_sum = final_summary_ollama_with_range(summaries, target_range=sentence_range)
            if not final_sum:
                raise Exception("Ollama returned empty final summary.")
            
            # Post-process to preserve amounts
            final_sum = preserve_amounts_in_summary(text, final_sum)
            return final_sum

        # REGULAR PATH (Pegasus/Finetuned)
        if active_model is None or active_tokenizer is None:
            raise Exception("Summarization models are not loaded. Please wait for models to initialize.")
        
        # Extract key sentences
        if use_minilm and minilm_tokenizer and minilm_encoder:
            ext = ' '.join(textrank_with_minilm(text, top_n=top_k_sent))
        else:
            ext = ' '.join(textrank_extract(text, top_n=top_k_sent))
        
        if not ext or len(ext.strip()) == 0:
            raise Exception("Could not extract key sentences from the text.")
        
        # Chunk with the tokenizer that matches the active summarizer
        chunks = chunk_text(ext, max_tokens=MAX_INPUT_TOKENS - 200, tokenizer=active_tokenizer)
        if not chunks or len(chunks) == 0:
            raise Exception("Could not chunk the extracted text.")
        
        joined = ' </s> '.join(chunks)

        focus = (theme_focus or 'general').strip().lower()
        if focus not in ALLOWED_THEME_FOCUS:
            focus = 'general'
        prefix = THEME_FOCUS_INSTRUCTIONS.get(focus, '')
        if prefix:
            joined = f'{prefix}\n\n{joined}'
        
        # Summarize using pipeline
        summary = None
        if active_pipe:
            # Try pipeline summarization
            try:
                print(f"[DEBUG] Input to pipeline length: {len(joined)}")
                print(f"[DEBUG] Input preview: {joined[:200]}...")
                
                # Try without min_length first, as it can cause empty outputs
                outputs = active_pipe(
                    joined, 
                    max_length=200,
                    min_length=10,   # Very low minimum to avoid empty summaries
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True
                )
                print(f"[DEBUG] Pipeline output type: {type(outputs)}, Length: {len(outputs) if outputs else 0}")
                print(f"[DEBUG] Pipeline output: {outputs}")
                
                if not outputs or len(outputs) == 0:
                    raise Exception("Pipeline returned empty output.")
                
                # Handle different output formats
                if isinstance(outputs, list) and len(outputs) > 0:
                    if isinstance(outputs[0], dict):
                        summary = outputs[0].get('summary_text', '')
                    elif isinstance(outputs[0], str):
                        summary = outputs[0]
                    else:
                        summary = str(outputs[0])
                elif isinstance(outputs, str):
                    summary = outputs
                else:
                    summary = str(outputs)
                
                print(f"[DEBUG] Extracted summary: {summary[:100] if summary else 'None'}...")
                
                # If summary is still empty, try without min_length constraint
                if not summary or len(summary.strip()) == 0:
                    print("[DEBUG] First attempt empty, trying without min_length...")
                    outputs = active_pipe(
                        joined,
                        max_length=200,
                        do_sample=False,
                        num_beams=4,
                        early_stopping=True
                    )
                    if isinstance(outputs, list) and len(outputs) > 0:
                        if isinstance(outputs[0], dict):
                            summary = outputs[0].get('summary_text', '')
                        elif isinstance(outputs[0], str):
                            summary = outputs[0]
                        else:
                            summary = str(outputs[0])
                    elif isinstance(outputs, str):
                        summary = outputs
                    else:
                        summary = str(outputs)
                    
                    print(f"[DEBUG] Second attempt summary: {summary[:100] if summary else 'None'}...")
                
                if not summary or len(summary.strip()) == 0:
                    print("[WARNING] Generated summary is empty after multiple attempts, will use fallback")
                    summary = None
                    
            except Exception as pipe_error:
                print(f"[ERROR] Pipeline error: {str(pipe_error)}")
                # Don't raise yet - try fallback to extracted sentences
                summary = None
        else:
            # Fallback to direct model call
            if active_model is None:
                raise Exception("Model is not loaded.")
            
            inputs = active_tokenizer(joined, return_tensors='pt', truncation=True, max_length=MAX_INPUT_TOKENS)
            inputs = {k: v.to(active_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = active_model.generate(
                    **inputs,
                    max_length=200,  # Increased to preserve amounts
                    min_length=30,   # Reduced from 50 to avoid empty summaries
                    do_sample=False,
                    num_beams=5
                )
            
            print(f"[DEBUG] Model generation output type: {type(outputs)}, Shape: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
            
            if outputs is None or len(outputs) == 0:
                raise Exception("Model generation returned empty output.")
            
            summary = active_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[DEBUG] Decoded summary: {summary[:100]}...")
            
            if not summary or len(summary.strip()) == 0:
                summary = None  # Set to None to trigger fallback
        
        # If both pipeline and model generation failed, use extracted sentences as fallback
        if not summary or len(summary.strip()) == 0:
            print("[WARNING] All summarization methods failed, using extracted sentences as fallback")
            # Use the extracted sentences as a simple summary
            if use_minilm and minilm_tokenizer and minilm_encoder:
                ext_sents = textrank_with_minilm(text, top_n=top_k_sent)
            else:
                ext_sents = textrank_extract(text, top_n=top_k_sent)
            summary = ' '.join(ext_sents)
            print(f"[DEBUG] Fallback summary length: {len(summary)}")
        
        # Post-process to preserve amounts
        summary = preserve_amounts_in_summary(text, summary)
        
        # Final validation
        if not summary or len(summary.strip()) == 0:
            raise Exception("Final summary is empty after processing.")
        
        return summary
    except Exception as e:
        error_msg = f"Error during summarization: {str(e)}"
        print(f"Summarization error: {error_msg}")
        return error_msg


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text to fix common extraction issues."""
    if not text:
        return text
    
    # Normalize Unicode characters (fix encoding issues)
    text = unicodedata.normalize('NFKC', text)
    
    # Fix common Telugu OCR/extraction mistakes
    telugu_fixes = {
        # Common character misrecognitions
        r'మొదంటి': 'మొదటి',  # first
        r'వాయిందా': 'వాయిదా',  # installment
        r'నుడిం': 'నుండి',  # from
        r'దంకుదురు': 'కుదుర్చుకున్న',  # made/agreed
        r'ఒప\s+దంకుదురు': 'ఒప్పందం కుదుర్చుకున్న',  # agreement made
        r'రాX': 'తర్వాత',  # after
        r'రా\s*X': 'తర్వాత',  # after (with space)
        r'([\u0C00-\u0C7F])\s*W\s*([\u0C00-\u0C7F])': r'\1\2',  # Remove stray W between Telugu chars
        r'([\u0C00-\u0C7F])\s*X\s*([\u0C00-\u0C7F\d])': r'\1\2',  # Remove stray X between Telugu chars/numbers
        # Fix spacing issues in Telugu
        r'(\S)\s+([ాీుూెేైొోౌృౄఁంః])': r'\1\2',  # Remove space before Telugu diacritics
        r'([ాీుూెేైొోౌృౄఁంః])\s+(\S)': r'\1\2',  # Remove space after Telugu diacritics
    }
    
    for pattern, replacement in telugu_fixes.items():
        text = re.sub(pattern, replacement, text)
    
    # Fix common spacing issues
    # Remove excessive whitespace but preserve line breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
    
    # Fix broken words (common in PDF extraction)
    # Join words that are split across lines incorrectly
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated line breaks
    
    # Fix common character recognition errors
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,;:!?])\s*([.,;:!?])', r'\1\2', text)  # Fix double punctuation spacing
    
    # CRITICAL: Preserve amounts with commas (Indian number format)
    # First, protect amounts by temporarily replacing them
    amount_pattern = r'రూ\.\s*([\d,]+)'
    amounts = re.findall(amount_pattern, text)
    amount_placeholders = {}
    for i, amount in enumerate(amounts):
        placeholder = f'__AMOUNT_{i}__'
        amount_placeholders[placeholder] = amount
        text = re.sub(r'రూ\.\s*' + re.escape(amount), f'రూ. {placeholder}', text, count=1)
    
    # Fix common OCR/extraction mistakes for numbers and currency
    # Join split numbers (but be careful with comma-separated numbers)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Join split numbers (but not comma-separated)
    text = re.sub(r'₹\s+(\d)', r'₹\1', text)  # Fix currency symbol spacing
    text = re.sub(r'Rs\.\s+(\d)', r'Rs. \1', text)  # Fix Rs. spacing
    
    # Restore protected amounts
    for placeholder, amount in amount_placeholders.items():
        text = text.replace(placeholder, amount)
    
    # Fix common word splitting issues
    # Join single characters that should be part of words (common in Indian languages)
    # Use Unicode character classes for better matching
    text = re.sub(r'([\u0C00-\u0C7F])\s+([ాీుూెేైొోౌృౄఁంః])\s+([\u0C00-\u0C7F])', r'\1\2\3', text)  # Telugu
    text = re.sub(r'([\u0900-\u097F])\s+([ाीुूेैोौृॄंः])\s+([\u0900-\u097F])', r'\1\2\3', text)  # Hindi
    text = re.sub(r'([\u0B80-\u0BFF])\s+([ாிீுூெேைொோௌ்ர்ல்])\s+([\u0B80-\u0BFF])', r'\1\2\3', text)  # Tamil
    
    # Fix spacing around Telugu words and numbers
    text = re.sub(r'రూ\.\s*([\d,]+)', r'రూ. \1', text)  # Ensure space after రూ.
    text = re.sub(r'([\d,]+)\s*-\s*([\u0C00-\u0C7F])', r'\1 - \2', text)  # Fix spacing around dashes with Telugu
    
    # Remove control characters but preserve newlines and tabs
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r')
    
    # Clean up the final text
    text = text.strip()
    
    return text


def _looks_like_table_noise_line(line: str) -> bool:
    """Detect low-value rows from tables/OCR before they pollute chunking."""
    s = re.sub(r'\s+', ' ', (line or '')).strip()
    if not s:
        return True
    if re.search(r'^[\d\s|:_./\\-]+$', s):
        return True

    alpha = len(re.findall(r'[A-Za-z\u0C00-\u0C7F\u0900-\u097F\u0B80-\u0BFF]', s))
    digits = len(re.findall(r'\d', s))
    alpha_density = alpha / max(len(s), 1)
    digit_density = digits / max(len(s), 1)
    token_count = len(s.split())

    if token_count <= 3 and digit_density > 0.25:
        return True
    if alpha_density < 0.26 and digit_density > 0.14:
        return True
    if s.count('|') >= 2 and alpha_density < 0.45:
        return True
    return False


def _clean_layout_text(text: str) -> str:
    """Preserve paragraph structure while removing noisy table-like lines."""
    if not text:
        return ''

    paras = re.split(r'\n\s*\n+', text)
    cleaned_paras = []
    for para in paras:
        lines = []
        for raw_line in para.splitlines():
            line = re.sub(r'\s+', ' ', raw_line).strip()
            if not line or _looks_like_table_noise_line(line):
                continue
            lines.append(line)
        if lines:
            cleaned_paras.append('\n'.join(lines))
    return '\n\n'.join(cleaned_paras).strip()


def _bbox_contains_word(bbox: tuple, word: dict, tolerance: float = 1.5) -> bool:
    x0, top, x1, bottom = bbox
    return (
        float(word.get('x0', 0.0)) >= x0 - tolerance and
        float(word.get('x1', 0.0)) <= x1 + tolerance and
        float(word.get('top', 0.0)) >= top - tolerance and
        float(word.get('bottom', 0.0)) <= bottom + tolerance
    )


def _extract_pdfplumber_structured(pdf_bytes: bytes) -> str:
    import pdfplumber

    pages_out = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            table_bboxes = []
            try:
                table_bboxes = [tbl.bbox for tbl in page.find_tables()]
            except Exception:
                table_bboxes = []

            page_words = []
            try:
                page_words = page.extract_words(
                    use_text_flow=True,
                    keep_blank_chars=False,
                    x_tolerance=2,
                    y_tolerance=3,
                )
            except Exception:
                page_words = []

            if page_words:
                kept = []
                for word in page_words:
                    if any(_bbox_contains_word(bbox, word) for bbox in table_bboxes):
                        continue
                    kept.append(word)

                lines = []
                current_words = []
                current_top = None
                for word in kept:
                    word_top = float(word.get('top', 0.0))
                    if current_top is None or abs(word_top - current_top) <= 3.0:
                        current_words.append(word)
                        current_top = word_top if current_top is None else min(current_top, word_top)
                    else:
                        line = ' '.join(w.get('text', '').strip() for w in sorted(current_words, key=lambda item: float(item.get('x0', 0.0))))
                        if line.strip():
                            lines.append(line.strip())
                        current_words = [word]
                        current_top = word_top
                if current_words:
                    line = ' '.join(w.get('text', '').strip() for w in sorted(current_words, key=lambda item: float(item.get('x0', 0.0))))
                    if line.strip():
                        lines.append(line.strip())

                page_text = _clean_layout_text('\n'.join(lines))
                if page_text:
                    pages_out.append(page_text)
                    continue

            page_text = ''
            try:
                page_text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=3) or ''
            except TypeError:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=3) or ''
            page_text = _clean_layout_text(page_text)
            if page_text:
                pages_out.append(page_text)

    return '\n\n'.join(pages_out).strip()


def _extract_pymupdf_structured(pdf_bytes: bytes) -> str:
    try:
        import fitz
    except ImportError:
        return ''

    pages_out = []
    with fitz.open(stream=pdf_bytes, filetype='pdf') as pdf:
        for page in pdf:
            blocks = []
            try:
                blocks = page.get_text('blocks')
            except Exception:
                blocks = []
            if not blocks:
                continue
            filtered = []
            for block in sorted(blocks, key=lambda item: (item[1], item[0])):
                text = re.sub(r'\s+', ' ', str(block[4] or '')).strip()
                if not text or _looks_like_table_noise_line(text):
                    continue
                filtered.append(text)
            page_text = _clean_layout_text('\n\n'.join(filtered))
            if page_text:
                pages_out.append(page_text)
    return '\n\n'.join(pages_out).strip()


def _extraction_quality_score(text: str) -> float:
    """Heuristic score to pick the least-corrupted extraction output."""
    if not text or not isinstance(text, str):
        return -1.0

    s = text.strip()
    if len(s) < 40:
        return -1.0

    total_len = len(s)
    alpha_chars = len(re.findall(r'[A-Za-z\u0C00-\u0C7F\u0900-\u097F\u0B80-\u0BFF]', s))
    alpha_density = alpha_chars / max(total_len, 1)

    words = re.findall(r'[A-Za-z\u0C00-\u0C7F\u0900-\u097F\u0B80-\u0BFF]+', s)
    if not words:
        return -1.0

    short_word_ratio = sum(1 for w in words if len(w) <= 2) / len(words)

    english_words = [w.lower() for w in re.findall(r'[A-Za-z]+', s)]
    no_vowel_ratio = 0.0
    if english_words:
        long_en_words = [w for w in english_words if len(w) >= 4]
        if long_en_words:
            no_vowel_ratio = sum(1 for w in long_en_words if not re.search(r'[aeiou]', w)) / len(long_en_words)

    weird_chars = len(re.findall(r'[^A-Za-z0-9\u0C00-\u0C7F\u0900-\u097F\u0B80-\u0BFF\s.,;:!?()\[\]{}"\'\-₹/%&]', s))
    weird_ratio = weird_chars / max(total_len, 1)

    return (
        alpha_density * 2.0
        - short_word_ratio * 1.2
        - no_vowel_ratio * 1.8
        - weird_ratio * 2.0
    )


def _best_extracted_text(candidates: list) -> tuple:
    """Return best (method_name, cleaned_text, score) from candidates."""
    best_method, best_text, best_score = None, "", -1.0
    for method_name, raw_text in candidates:
        cleaned = clean_extracted_text(raw_text or "")
        score = _extraction_quality_score(cleaned)
        if score > best_score:
            best_method, best_text, best_score = method_name, cleaned, score
    return best_method, best_text, best_score


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file with improved accuracy and text cleaning."""
    original_position = pdf_file.tell() if hasattr(pdf_file, 'tell') else None
    methods_tried = []
    candidates = []

    try:
        if hasattr(pdf_file, 'seek'):
            pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        if not pdf_bytes:
            raise ValueError("Uploaded PDF is empty.")
    finally:
        if hasattr(pdf_file, 'seek') and original_position is not None:
            pdf_file.seek(original_position)

    # Method 1: PyMuPDF structured blocks (best when available)
    try:
        method_text = _extract_pymupdf_structured(pdf_bytes)
        methods_tried.append("PyMuPDF")
        if method_text:
            candidates.append(("PyMuPDF", method_text))
    except Exception as e:
        methods_tried.append(f"PyMuPDF (failed: {str(e)})")

    # Method 2: pdfplumber with table filtering + layout-aware reconstruction
    try:
        method_text = _extract_pdfplumber_structured(pdf_bytes)
        methods_tried.append("pdfplumber_structured")
        if method_text:
            candidates.append(("pdfplumber_structured", method_text))
    except ImportError:
        methods_tried.append("pdfplumber (not installed)")
    except Exception as e:
        methods_tried.append(f"pdfplumber (failed: {str(e)})")

    # Method 3: pdfminer.six
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        method_text = _clean_layout_text(pdfminer_extract(BytesIO(pdf_bytes)))
        methods_tried.append("pdfminer")
        if method_text and method_text.strip():
            candidates.append(("pdfminer", method_text))
    except ImportError:
        methods_tried.append("pdfminer (not installed)")
    except Exception as e:
        methods_tried.append(f"pdfminer (failed: {str(e)})")

    # Method 4: PyPDF2
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        parts = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if not page_text or len(page_text.strip()) < 10:
                try:
                    page_text = page.extract_text(layout=True)
                except Exception:
                    pass
            if page_text:
                cleaned_page = _clean_layout_text(page_text)
                if cleaned_page:
                    parts.append(cleaned_page)
        methods_tried.append("PyPDF2")
        method_text = "\n\n".join(parts).strip()
        if method_text:
            candidates.append(("PyPDF2", method_text))
    except Exception as e:
        methods_tried.append(f"PyPDF2 (failed: {str(e)})")

    best_method, best_text, best_score = _best_extracted_text(candidates)
    if best_text and best_score >= 0.15:
        print(f"[DEBUG] PDF extraction selected: {best_method} (score={best_score:.3f})")
        text = best_text.strip()
        print(text[:5000])
        return text

    # Method 5: OCR fallback for scanned PDFs (optional dependencies)
    try:
        from pdf2image import convert_from_bytes
        import pytesseract

        images = convert_from_bytes(pdf_bytes, dpi=300)
        ocr_parts = []
        for image in images:
            page_text = pytesseract.image_to_string(image, lang='eng')
            if page_text:
                ocr_parts.append(page_text)
        ocr_text = "\n".join(ocr_parts).strip()
        if ocr_text:
            candidates.append(("ocr_tesseract", ocr_text))
    except Exception as e:
        methods_tried.append(f"OCR fallback (failed/unavailable: {str(e)})")

    best_method, best_text, best_score = _best_extracted_text(candidates)
    if best_text:
        print(f"[DEBUG] PDF extraction selected after OCR: {best_method} (score={best_score:.3f})")
        text = best_text.strip()
        print(text[:5000])
        return text

    error_msg = f"Error extracting text from PDF using methods: {', '.join(methods_tried) if methods_tried else 'none'}."
    raise Exception(error_msg)


def load_models():
    """Load all required models (Pegasus always; fine-tuned summarizer if present)."""
    global model_finetuned, tokenizer_finetuned, summarizer_pipe_finetuned
    global model_pegasus, tokenizer_pegasus, summarizer_pipe_pegasus
    global minilm_tokenizer, minilm_encoder
    
    print("Loading models...")
    
    try:
        _pipe_device = 0 if torch.cuda.is_available() else -1

        # Pegasus (google/pegasus-xsum) — always loaded for abstractive summarization option
        # print(f"Loading Pegasus summarizer: {SUMMARIZER}")
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER)
        tokenizer_pegasus = AutoTokenizer.from_pretrained(SUMMARIZER)
        model_pegasus.to(device)
        model_pegasus.eval()
        summarizer_pipe_pegasus = pipeline(
            'summarization',
            model=model_pegasus,
            tokenizer=tokenizer_pegasus,
            device=_pipe_device,
        )

        # Optional domain-specific fine-tuned seq2seq (same API as Pegasus)
        model_path = os.environ.get('DOMAIN_SUMMARIZER_PATH', './scientific_summarizer_final')
        if os.path.exists(model_path):
            try:
                print(f"Loading fine-tuned summarizer from {model_path}")
                model_finetuned = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                tokenizer_finetuned = AutoTokenizer.from_pretrained(model_path)
                model_finetuned.to(device)
                model_finetuned.eval()
                summarizer_pipe_finetuned = pipeline(
                    'summarization',
                    model=model_finetuned,
                    tokenizer=tokenizer_finetuned,
                    device=_pipe_device,
                )
            except Exception as fin_err:
                print(f"Warning: could not load fine-tuned model from {model_path}: {fin_err}")
                model_finetuned = None
                tokenizer_finetuned = None
                summarizer_pipe_finetuned = None
        else:
            print(f"No folder {model_path} — only Pegasus will be used for summarization.")
            model_finetuned = None
            tokenizer_finetuned = None
            summarizer_pipe_finetuned = None
        
        # Load MiniLM for domain adaptation
        print(f"Loading domain encoder: {DOMAIN_ENCODER}")
        minilm_tokenizer = AutoTokenizer.from_pretrained(DOMAIN_ENCODER)
        minilm_encoder = AutoModel.from_pretrained(DOMAIN_ENCODER)
        minilm_encoder.to(device)
        minilm_encoder.eval()
        
        print("All models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    """API endpoint for summarization."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        top_k_sent, sentence_range = parse_top_k_option(data.get('top_k_sent', 8))
        use_minilm = data.get('use_minilm', True)
        use_pegasus = bool(data.get('use_pegasus', False))
        use_ollama = bool(data.get('use_ollama', True))
        summary_format, theme_focus = normalize_summary_options(
            data.get('summary_format', 'paragraphs'),
            data.get('theme_focus', 'general'),
        )
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Please provide text to summarize.'
            }), 400
        
        if len(text) > 50000:  # Limit text length
            return jsonify({
                'success': False,
                'error': 'Text is too long. Maximum 50,000 characters allowed.'
            }), 400
        
        input_language_code, input_language_name = detect_input_language(text)

        # Generate summary
        _a, _b, _c, summarizer_backend = _pick_summarizer_backend(use_pegasus, use_ollama)
        summary = summarize_scientific(
            text,
            top_k_sent=top_k_sent,
            use_minilm=use_minilm,
            theme_focus=theme_focus,
            use_pegasus=use_pegasus,
            use_ollama=use_ollama,
            sentence_range=sentence_range,
        )
        summary_display = format_summary_display(summary, summary_format)

        (
            source_summary,
            source_summary_display,
            english_summary,
            english_summary_display,
            summary_display,
            bilingual,
        ) = build_bilingual_summaries(
            summary=summary,
            summary_format=summary_format,
            input_language_code=input_language_code,
            input_language_name=input_language_name,
        )
        
        # Validate summary before proceeding
        if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Summary generation failed. The model may not be loaded properly or the text could not be processed. Please check the server logs for details.'
            }), 500
        
        # Check if summary is an error message
        if summary.startswith('Error during summarization:'):
            return jsonify({
                'success': False,
                'error': summary
            }), 500
        
        # Get extracted sentences for display
        if use_minilm and minilm_tokenizer and minilm_encoder:
            extracted_sents = textrank_with_minilm(text, top_n=top_k_sent)
        else:
            extracted_sents = textrank_extract(text, top_n=top_k_sent)
        
        # Preserve amounts in extracted sentences
        extracted_sents = preserve_amounts_in_sentences(text, extracted_sents)
        
        # Compute LIME and SHAP explanations
        lime_scores = compute_sentence_importance_lime(text, summary, top_k_sent, use_minilm)
        shap_values = compute_shap_values_simple(text, summary, top_k_sent, use_minilm)
        
        # Prepare sentence data with explanations
        sentences = sent_tokenize(text)
        sentence_explanations = []
        for i, sent in enumerate(sentences[:50]):  # Limit to first 50 sentences
            sentence_explanations.append({
                'sentence': sent[:200] + '...' if len(sent) > 200 else sent,  # Truncate long sentences
                'lime_score': round(lime_scores.get(i, 0.0), 3),
                'shap_value': round(shap_values.get(i, 0.0), 3),
                'index': i
            })
        
        return jsonify({
            'success': True,
            'summary': summary,
            'summary_display': summary_display,
            'summary_input_language': source_summary,
            'summary_input_language_display': source_summary_display,
            'summary_english': english_summary,
            'summary_english_display': english_summary_display,
            'is_bilingual': bilingual,
            'input_language_code': input_language_code,
            'input_language_name': input_language_name,
            'summary_format': summary_format,
            'theme_focus': theme_focus,
            'summarizer_backend': summarizer_backend,
            'use_pegasus': use_pegasus,
            'extracted_sentences': extracted_sents,
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': f"{(1 - len(summary)/len(text))*100:.1f}%",
            'explanations': {
                'lime_scores': sentence_explanations,
                'shap_values': sentence_explanations
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error during summarization: {str(e)}'
        }), 500


@app.route('/summarize-pdf', methods=['POST'])
def summarize_pdf():
    """API endpoint for PDF upload and summarization."""
    try:
        # Check if PDF file is present
        if 'pdf' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No PDF file provided.'
            }), 400
        
        pdf_file = request.files['pdf']
        
        if pdf_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected.'
            }), 400
        
        if not allowed_file(pdf_file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Only PDF files are allowed.'
            }), 400
        
        # Get parameters
        top_k_sent, sentence_range = parse_top_k_option(request.form.get('top_k_sent', '6-8'))
        use_minilm = request.form.get('use_minilm', 'true').lower() == 'true'
        use_pegasus = request.form.get('use_pegasus', 'false').lower() == 'true'
        use_ollama = request.form.get('use_ollama', 'true').lower() == 'true'
        summary_format, theme_focus = normalize_summary_options(
            request.form.get('summary_format', 'paragraphs'),
            request.form.get('theme_focus', 'general'),
        )
        
        # Extract text from PDF
        try:
            text = extract_text_from_pdf(pdf_file)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'No text could be extracted from the PDF.'
            }), 400
        
        # Limit text length (truncate if too long)
        if len(text) > 50000:
            text = text[:50000]
            # Try to truncate at a sentence boundary
            last_period = text.rfind('.')
            if last_period > 40000:  # Only truncate at sentence if reasonable
                text = text[:last_period + 1]
        
        input_language_code, input_language_name = detect_input_language(text)

        # Generate summary
        print(f"[DEBUG] Generating summary for text length: {len(text)}")
        _a, _b, _c, summarizer_backend = _pick_summarizer_backend(use_pegasus, use_ollama)
        summary = summarize_scientific(
            text,
            top_k_sent=top_k_sent,
            use_minilm=use_minilm,
            theme_focus=theme_focus,
            use_pegasus=use_pegasus,
            use_ollama=use_ollama,
            sentence_range=sentence_range,
        )
        summary_display = format_summary_display(summary, summary_format)

        (
            source_summary,
            source_summary_display,
            english_summary,
            english_summary_display,
            summary_display,
            bilingual,
        ) = build_bilingual_summaries(
            summary=summary,
            summary_format=summary_format,
            input_language_code=input_language_code,
            input_language_name=input_language_name,
        )
        print(f"[DEBUG] Summary generated. Type: {type(summary)}, Length: {len(summary) if summary else 0}")
        print(f"[DEBUG] Summary content: {summary[:200] if summary else 'None'}...")
        
        # Validate summary before proceeding
        if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Summary generation failed. The model may not be loaded properly or the text could not be processed. Please check the server logs for details.'
            }), 500
        
        # Check if summary is an error message
        if summary.startswith('Error during summarization:'):
            return jsonify({
                'success': False,
                'error': summary
            }), 500
        
        # Get extracted sentences for display
        if use_minilm and minilm_tokenizer and minilm_encoder:
            extracted_sents = textrank_with_minilm(text, top_n=top_k_sent)
        else:
            extracted_sents = textrank_extract(text, top_n=top_k_sent)
        
        # Preserve amounts in extracted sentences
        extracted_sents = preserve_amounts_in_sentences(text, extracted_sents)
        
        # Compute LIME and SHAP explanations
        lime_scores = compute_sentence_importance_lime(text, summary, top_k_sent, use_minilm)
        shap_values = compute_shap_values_simple(text, summary, top_k_sent, use_minilm)
        
        # Prepare sentence data with explanations
        sentences = sent_tokenize(text)
        sentence_explanations = []
        for i, sent in enumerate(sentences[:50]):  # Limit to first 50 sentences
            sentence_explanations.append({
                'sentence': sent[:200] + '...' if len(sent) > 200 else sent,
                'lime_score': round(lime_scores.get(i, 0.0), 3),
                'shap_value': round(shap_values.get(i, 0.0), 3),
                'index': i
            })
        
        return jsonify({
            'success': True,
            'summary': summary,
            'summary_display': summary_display,
            'summary_input_language': source_summary,
            'summary_input_language_display': source_summary_display,
            'summary_english': english_summary,
            'summary_english_display': english_summary_display,
            'is_bilingual': bilingual,
            'input_language_code': input_language_code,
            'input_language_name': input_language_name,
            'summary_format': summary_format,
            'theme_focus': theme_focus,
            'summarizer_backend': summarizer_backend,
            'use_pegasus': use_pegasus,
            'extracted_sentences': extracted_sents,
            'extracted_text': text,  # Include extracted text for verification
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': f"{(1 - len(summary)/len(text))*100:.1f}%",
            'pdf_filename': secure_filename(pdf_file.filename),
            'explanations': {
                'lime_scores': sentence_explanations,
                'shap_values': sentence_explanations
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error during PDF processing: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    models_loaded = model_pegasus is not None and tokenizer_pegasus is not None
    return jsonify({
        'status': 'healthy' if models_loaded else 'models_not_loaded',
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'models_loaded': models_loaded,
        'pegasus_loaded': model_pegasus is not None,
        'finetuned_summarizer_loaded': model_finetuned is not None,
    })




@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            session['username'] = mail1
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            session['username'] = mail1
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")


@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/summary')
def summary():
    return render_template('summary.html')

@app.route("/chat")
def chat():
    return render_template("chat.html")


def _extract_chat_pdf_text(pdf_data: str) -> tuple:
    """
    Decode base64 / data-URL PDF and return extracted text.
    Returns (True, text) on success or (False, error_message) on failure.
    """
    if not pdf_data or not isinstance(pdf_data, str):
        return False, 'No PDF data provided.'
    if 'base64,' in pdf_data:
        pdf_data = pdf_data.split('base64,', 1)[-1]
    pdf_data = pdf_data.strip()
    try:
        raw = base64.b64decode(pdf_data, validate=False)
    except Exception as e:
        return False, f'Could not decode PDF data: {e}'
    if not raw:
        return False, 'Could not decode PDF data.'
    try:
        text = extract_text_from_pdf(BytesIO(raw))
    except Exception as e:
        return False, str(e)
    if not text or not str(text).strip():
        return False, 'No text could be extracted from this PDF. It may be scanned or image-only.'
    text = str(text).strip()
    if len(text) > 50000:
        text = text[:50000]
    return True, text


@app.route('/chat-prepare', methods=['POST'])
def chat_prepare():
    """Extract plain text from a PDF (base64) for chat / RAG. Called once when opening chat with a PDF."""
    try:
        data = request.get_json() or {}
        pdf_data = data.get('pdf_data') or ''
        ok, payload = _extract_chat_pdf_text(pdf_data)
        if not ok:
            return jsonify({'success': False, 'error': payload}), 400
        return jsonify({'success': True, 'extracted_text': payload, 'length': len(payload)})
    except Exception as e:
        return jsonify({'success': False, 'error': f'PDF preparation failed: {str(e)}'}), 500


@app.route('/chat-history/conversations', methods=['GET'])
def chat_history_conversations():
    username = _current_username()
    if not username:
        return jsonify({'success': False, 'error': 'Please sign in to access chat history.'}), 401
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, title, created_at, updated_at
        FROM chat_conversations
        WHERE username = ?
        ORDER BY updated_at DESC, id DESC
        """,
        (username,),
    )
    rows = cur.fetchall()
    con.close()
    items = [
        {
            'id': int(r[0]),
            'title': r[1],
            'created_at': r[2],
            'updated_at': r[3],
        }
        for r in rows
    ]
    return jsonify({'success': True, 'conversations': items})


@app.route('/chat-history/conversations', methods=['POST'])
def chat_history_create_conversation():
    username = _current_username()
    if not username:
        return jsonify({'success': False, 'error': 'Please sign in to create a conversation.'}), 401
    data = request.get_json() or {}
    first_question = (data.get('first_question') or '').strip()
    conv_id = _create_conversation(username, first_question=first_question)
    return jsonify({'success': True, 'conversation_id': conv_id})


@app.route('/chat-history/conversations/<int:conversation_id>/messages', methods=['GET'])
def chat_history_messages(conversation_id: int):
    username = _current_username()
    if not username:
        return jsonify({'success': False, 'error': 'Please sign in to access chat history.'}), 401
    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT id FROM chat_conversations WHERE id = ? AND username = ?",
        (conversation_id, username),
    )
    owner = cur.fetchone()
    if not owner:
        con.close()
        return jsonify({'success': False, 'error': 'Conversation not found.'}), 404
    cur.execute(
        """
        SELECT role, content, created_at
        FROM chat_messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,),
    )
    rows = cur.fetchall()
    con.close()
    messages = [{'role': r[0], 'content': r[1], 'created_at': r[2]} for r in rows]
    return jsonify({'success': True, 'messages': messages})


@app.route('/chat-history/conversations/<int:conversation_id>', methods=['DELETE'])
def chat_history_delete_conversation(conversation_id: int):
    username = _current_username()
    if not username:
        return jsonify({'success': False, 'error': 'Please sign in to manage chat history.'}), 401

    con = _db_connect()
    cur = con.cursor()
    cur.execute(
        "SELECT id FROM chat_conversations WHERE id = ? AND username = ?",
        (conversation_id, username),
    )
    owner = cur.fetchone()
    if not owner:
        con.close()
        return jsonify({'success': False, 'error': 'Conversation not found.'}), 404

    cur.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
    cur.execute("DELETE FROM chat_conversations WHERE id = ? AND username = ?", (conversation_id, username))
    con.commit()
    con.close()
    return jsonify({'success': True})


@app.route('/chat-query', methods=['POST'])
def chat_query():
    """RAG-backed answers about the active scientific document (see scientific_rag.py)."""
    try:
        data = request.get_json() or {}
        question = (data.get('question') or '').strip()
        doc_text = (data.get('document_text') or '').strip()
        pdf_data = data.get('pdf_data')
        history = data.get('history')
        conversation_id = data.get('conversation_id')
        username = _current_username()

        if not question:
            return jsonify({'success': False, 'error': 'Please enter a question.'}), 400
        if not username:
            return jsonify({'success': False, 'error': 'Please sign in to chat.'}), 401

        if len(doc_text) < 30 and pdf_data:
            ok, payload = _extract_chat_pdf_text(pdf_data)
            if not ok:
                return jsonify({'success': False, 'error': payload}), 400
            doc_text = payload.strip()

        if len(doc_text) < 30:
            return jsonify({
                'success': False,
                'error': 'No document text loaded. Go back to Home, upload a PDF, then open Chat again.',
            }), 400

        hist = None
        if isinstance(history, list):
            cleaned = []
            for t in history[-8:]:
                if not isinstance(t, dict):
                    continue
                r, c = t.get('role'), t.get('content')
                if r in ('user', 'assistant') and isinstance(c, str) and c.strip():
                    cleaned.append({'role': r, 'content': c[:8000]})
            if cleaned:
                hist = cleaned

        con = _db_connect()
        cur = con.cursor()
        conv_id = None
        if isinstance(conversation_id, int):
            cur.execute(
                "SELECT id FROM chat_conversations WHERE id = ? AND username = ?",
                (conversation_id, username),
            )
            if cur.fetchone():
                conv_id = int(conversation_id)
        else:
            try:
                cid = int(str(conversation_id).strip()) if conversation_id is not None else None
                if cid:
                    cur.execute(
                        "SELECT id FROM chat_conversations WHERE id = ? AND username = ?",
                        (cid, username),
                    )
                    if cur.fetchone():
                        conv_id = cid
            except Exception:
                conv_id = None
        con.close()
        if conv_id is None:
            conv_id = _create_conversation(username, first_question=question)
        else:
            con = _db_connect()
            cur = con.cursor()
            cur.execute(
                "SELECT COUNT(1) FROM chat_messages WHERE conversation_id = ?",
                (conv_id,),
            )
            msg_count = cur.fetchone()[0] or 0
            if int(msg_count) == 0:
                title = question[:67].rstrip() + '...' if len(question) > 70 else question
                if title:
                    cur.execute(
                        "UPDATE chat_conversations SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (title, conv_id),
                    )
                    con.commit()
            con.close()

        _save_message(conv_id, 'user', question)
        answer = rag_answer(doc_text, question, history=hist)
        _save_message(conv_id, 'assistant', answer)
        return jsonify({'success': True, 'answer': answer, 'conversation_id': conv_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


if __name__ == '__main__':
    # Load models on startup
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if load_models():
        print("\nStarting Flask server...")
        app.run(debug=False)
    else:
        print("\nFailed to load models. Please check the error messages above.")
        print("The application will still start but summarization may not work.")
