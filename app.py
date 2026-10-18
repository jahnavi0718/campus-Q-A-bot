"""
=============================================================================
 Campus Q&A Bot - Lightweight Student RAG Application
=============================================================================
 A simple, robust Retrieval-Augmented Generation (RAG) backend built with Flask.
 Students upload PDF lecture notes, ask questions, and receive structured answers
 with exact page citations, key bullet points, and high-level summaries.

 Tech Stack:
  - Framework: Flask + Flask-CORS
  - PDF Extraction: pypdf (page-by-page extraction)
  - Embedding Model: sentence-transformers (all-MiniLM-L6-v2)
  - Vector Search: NumPy (In-memory cosine similarity, zero external database)
  - Generation: Google GenAI SDK (Gemini Flash) with smart extractive fallback
=============================================================================
"""

import os
import re
import json
import random
from typing import List, Dict, Any, Optional
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Initialize Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ---------------------------------------------------------------------------
# Configuration & Global State
# ---------------------------------------------------------------------------
CHUNK_SIZE_WORDS = 120       # Target size per chunk in words (~120 words)
CHUNK_OVERLAP_WORDS = 20     # Overlap between consecutive chunks (~20 words)
MIN_SIMILARITY_THRESHOLD = 0.30  # Fallback to "Not found" if similarity < 0.30

# Initialize Embedding Model (sentence-transformers: all-MiniLM-L6-v2)
# Produces 384-dimensional dense semantic vectors
print("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

# In-memory document storage:
# Each item in CHUNKS is a dict:
# {
#     "id": int,
#     "file": str,
#     "subject": str,
#     "page": int,
#     "text": str,
#     "embedding": np.ndarray (shape 384,)
# }
CHUNKS: List[Dict[str, Any]] = []

# Optional Google GenAI SDK Client Initialization
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
gemini_client = None

if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini API client initialized successfully.")
    except Exception as e:
        print(f"Warning: Could not initialize Gemini client: {e}")
        gemini_client = None
else:
    print("Notice: GEMINI_API_KEY not found in environment. Extractive mode will be used.")


# ---------------------------------------------------------------------------
# Helper Functions: Chunking, Cosine Similarity, Extractive Fallback
# ---------------------------------------------------------------------------

def chunk_text_by_words(text: str, chunk_size: int = 120, overlap: int = 20) -> List[str]:
    """
    Splits raw text into overlapping word chunks of approximately `chunk_size` words,
    with `overlap` words carried over into the next chunk.
    """
    words = text.split()
    if not words:
        return []
    
    chunks = []
    step = max(1, chunk_size - overlap)
    
    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        # Ignore small tail chunks (less than 15 words) if we already have chunks
        if len(chunk_words) < 15 and len(chunks) > 0:
            continue
        chunks.append(" ".join(chunk_words))
    
    return chunks


def compute_cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a 1D query vector and a 2D matrix of document vectors.
    Returns a 1D array of similarity values in [-1, 1].
    """
    # Dot product between doc vectors and query vector
    dot_products = np.dot(doc_vecs, query_vec)
    # Norms
    norm_docs = np.linalg.norm(doc_vecs, axis=1)
    norm_query = np.linalg.norm(query_vec)
    
    # Avoid division by zero
    denominator = norm_docs * norm_query
    denominator = np.where(denominator == 0, 1e-9, denominator)
    
    return dot_products / denominator


def split_into_sentences(text: str) -> List[str]:
    """Splits a body of text into clean, trimmed sentences."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for s in raw_sentences:
        s_clean = s.strip()
        if len(s_clean) > 20 and not s_clean.startswith(('http', 'Page ', 'www.')):
            cleaned.append(s_clean)
    return cleaned


def extractive_fallback_answer(chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Extractive fallback if Gemini API key is missing or call fails.
    Extracts the most relevant sentences directly from retrieved chunks,
    ensuring the application NEVER breaks during a demo.
    """
    query_words = set(re.findall(r'\w+', query.lower()))
    candidate_sentences = []
    
    for c in chunks:
        sentences = split_into_sentences(c["text"])
        for s in sentences:
            s_words = set(re.findall(r'\w+', s.lower()))
            overlap = len(query_words.intersection(s_words))
            candidate_sentences.append({
                "sentence": s,
                "score": overlap,
                "file": c["file"],
                "page": c["page"]
            })
            
    # Sort sentences by keyword relevance
    candidate_sentences.sort(key=lambda x: x["score"], reverse=True)
    
    # Pick top 4-5 distinct sentences
    selected_points = []
    seen = set()
    for item in candidate_sentences:
        s_text = item["sentence"]
        if s_text not in seen:
            seen.add(s_text)
            selected_points.append(s_text)
        if len(selected_points) >= 5:
            break
            
    if not selected_points:
        # Fallback to chunk snippets
        selected_points = [c["text"][:180] + "..." for c in chunks[:4]]
        
    summary = selected_points[0] if selected_points else "Extracted relevant information from the uploaded notes."
    
    return {
        "points": selected_points[:5],
        "summary": summary
    }


def call_gemini_rag(context: str, question: str) -> Optional[Dict[str, Any]]:
    """
    Calls Google GenAI Gemini Flash model with a strict prompt requiring JSON output:
    { "points": [4-6 bullet points], "summary": "1-2 lines" }
    Falls back gracefully to None if any error occurs.
    """
    global gemini_client
    if not gemini_client:
        return None

    prompt = f"""You are an expert AI teaching assistant helping university students study.
Answer the student's question based ONLY on the provided context excerpts from their lecture notes.
Rules:
1. Explain clearly in simple, student-friendly language.
2. Rely strictly on the provided context. Do NOT fabricate or assume facts outside the text.
3. You MUST respond with ONLY valid JSON strictly matching this schema:
{{
  "points": [
    "First clear bullet point",
    "Second clear bullet point",
    "Third clear bullet point",
    "Fourth clear bullet point"
  ],
  "summary": "A concise 1 to 2 line summary capturing the core takeaway."
}}
Include between 4 and 6 informative bullet points in "points".

--- CONTEXT EXCERPTS ---
{context}

--- STUDENT QUESTION ---
{question}
"""

    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for model_name in models_to_try:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            raw_text = response.text.strip()
            
            # Clean possible markdown wrapping like ```json ... ```
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
                
            parsed = json.loads(raw_text)
            if "points" in parsed and "summary" in parsed and isinstance(parsed["points"], list):
                return parsed
        except Exception as e:
            print(f"Gemini call with {model_name} failed: {e}")
            continue

    return None


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serves the main single-page UI."""
    return render_template("index.html")


@app.route("/status", methods=["GET"])
def get_status():
    """
    Returns high-level statistics for the sidebar:
    total chunks, unique files, and list of subjects.
    """
    unique_files = list({c["file"] for c in CHUNKS})
    unique_subjects = sorted(list({c["subject"] for c in CHUNKS}))
    return jsonify({
        "chunks": len(CHUNKS),
        "files": len(unique_files),
        "file_names": unique_files,
        "subjects": unique_subjects,
        "gemini_active": gemini_client is not None
    })


@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Accepts multiple PDF files and a subject name.
    Extracts text page by page, splits into ~120 word chunks with 20 words overlap,
    computes embeddings with all-MiniLM-L6-v2, and stores in-memory.
    """
    subject = request.form.get("subject", "General").strip() or "General"
    uploaded_files = request.files.getlist("files")
    
    if not uploaded_files or len(uploaded_files) == 0:
        return jsonify({"error": "No files uploaded"}), 400

    new_chunks = []
    processed_files = []

    for file_storage in uploaded_files:
        filename = file_storage.filename or "Document.pdf"
        if not filename.lower().endswith(".pdf"):
            continue

        try:
            reader = PdfReader(file_storage)
            file_chunk_count = 0

            for page_idx, page in enumerate(reader.pages):
                page_number = page_idx + 1  # 1-indexed page number
                extracted_text = page.extract_text() or ""
                cleaned_text = re.sub(r'\s+', ' ', extracted_text).strip()
                
                if not cleaned_text:
                    continue
                
                # Split page into word chunks
                chunks = chunk_text_by_words(
                    cleaned_text,
                    chunk_size=CHUNK_SIZE_WORDS,
                    overlap=CHUNK_OVERLAP_WORDS
                )
                
                for text_chunk in chunks:
                    chunk_obj = {
                        "id": len(CHUNKS) + len(new_chunks),
                        "file": filename,
                        "subject": subject,
                        "page": page_number,
                        "text": text_chunk,
                        "embedding": None
                    }
                    new_chunks.append(chunk_obj)
                    file_chunk_count += 1
            
            processed_files.append({"name": filename, "chunks": file_chunk_count})

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return jsonify({"error": f"Failed to process {filename}: {str(e)}"}), 500

    if not new_chunks:
        return jsonify({"error": "Could not extract readable text from uploaded PDFs."}), 400

    # Compute dense embeddings for all new chunks in batch
    texts_to_embed = [c["text"] for c in new_chunks]
    embeddings = embedder.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=False)
    
    for idx, emb in enumerate(embeddings):
        new_chunks[idx]["embedding"] = emb

    # Add to global in-memory repository
    CHUNKS.extend(new_chunks)

    unique_subjects = sorted(list({c["subject"] for c in CHUNKS}))
    unique_files = list({c["file"] for c in CHUNKS})

    return jsonify({
        "success": True,
        "message": f"Successfully indexed {len(processed_files)} file(s) into {len(new_chunks)} chunks.",
        "files_processed": processed_files,
        "chunks_added": len(new_chunks),
        "total_chunks": len(CHUNKS),
        "total_files": len(unique_files),
        "subjects": unique_subjects
    })


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    RAG Question Answering Endpoint:
    1. Embeds question with all-MiniLM-L6-v2.
    2. Filters chunks by subject if requested.
    3. Retrieves top 4 chunks using NumPy cosine similarity.
    4. If similarity < 0.30, returns not found message.
    5. Formats context and prompts Gemini (or extractive fallback).
    6. Returns points, summary, deduplicated sources (file + page), confidence %.
    """
    data = request.get_json(silent=True) or request.form
    question = data.get("question", "").strip()
    selected_subject = data.get("subject", "All").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    if not CHUNKS:
        return jsonify({
            "points": ["No documents have been uploaded yet. Please upload PDF lecture notes using the sidebar."],
            "summary": "No documents uploaded.",
            "sources": [],
            "confidence": 0,
            "not_found": True
        })

    # Filter candidate chunks by subject
    if selected_subject and selected_subject.lower() != "all":
        candidate_chunks = [c for c in CHUNKS if c["subject"].lower() == selected_subject.lower()]
        if not candidate_chunks:
            # Fall back to all chunks if selected subject has no chunks
            candidate_chunks = CHUNKS
    else:
        candidate_chunks = CHUNKS

    # Embed user question
    query_vec = embedder.encode(question, convert_to_numpy=True)

    # Compute cosine similarities
    doc_vecs = np.array([c["embedding"] for c in candidate_chunks])
    sims = compute_cosine_similarity(query_vec, doc_vecs)

    # Rank and retrieve top 4
    top_k = min(4, len(candidate_chunks))
    top_indices = np.argsort(sims)[::-1][:top_k]
    
    best_sim = float(sims[top_indices[0]])
    confidence_pct = max(0, min(100, int(round(best_sim * 100))))
    
    # Threshold check: If best similarity is below 0.30 (30%), document does not have the answer
    if best_sim < MIN_SIMILARITY_THRESHOLD:
        return jsonify({
            "points": ["The uploaded notes do not seem to contain enough relevant information to answer this question."],
            "summary": "Not found in the uploaded documents.",
            "sources": [],
            "confidence": confidence_pct,
            "not_found": True
        })

    retrieved_chunks = [candidate_chunks[idx] for idx in top_indices]

    # Format deduplicated sources: "DBMS_Unit3.pdf · p.14"
    sources_seen = set()
    sources = []
    for c in retrieved_chunks:
        source_label = f"{c['file']} · p.{c['page']}"
        if source_label not in sources_seen:
            sources_seen.add(source_label)
            sources.append(source_label)

    # Prepare context text for LLM
    context_blocks = []
    for idx, c in enumerate(retrieved_chunks, 1):
        context_blocks.append(f"[Chunk {idx} - Source: {c['file']} Page: {c['page']}]\n{c['text']}")
    context_str = "\n\n".join(context_blocks)

    # Generation via Gemini API or fallback
    llm_result = call_gemini_rag(context_str, question)
    if not llm_result:
        llm_result = extractive_fallback_answer(retrieved_chunks, question)

    return jsonify({
        "points": llm_result.get("points", []),
        "summary": llm_result.get("summary", ""),
        "sources": sources,
        "confidence": confidence_pct,
        "not_found": False
    })


@app.route("/summary", methods=["POST"])
def generate_summary():
    """
    Generates a structured key points summary of notes for a selected subject (or all notes).
    """
    data = request.get_json(silent=True) or request.form
    selected_subject = data.get("subject", "All").strip()

    if not CHUNKS:
        return jsonify({
            "error": "No documents uploaded. Please upload PDFs first."
        }), 400

    if selected_subject and selected_subject.lower() != "all":
        candidate_chunks = [c for c in CHUNKS if c["subject"].lower() == selected_subject.lower()]
        if not candidate_chunks:
            candidate_chunks = CHUNKS
            selected_subject = "All Notes"
    else:
        candidate_chunks = CHUNKS
        selected_subject = "All Notes"

    # Select representative chunks across the subject (up to 6)
    if len(candidate_chunks) <= 6:
        selected_chunks = candidate_chunks
    else:
        # Uniform sampling across the document chunks
        indices = np.linspace(0, len(candidate_chunks) - 1, 6, dtype=int)
        selected_chunks = [candidate_chunks[i] for i in indices]

    sources_seen = set()
    sources = []
    for c in selected_chunks:
        lbl = f"{c['file']} · p.{c['page']}"
        if lbl not in sources_seen:
            sources_seen.add(lbl)
            sources.append(lbl)

    combined_text = "\n\n".join([f"Excerpt from {c['file']} (p.{c['page']}):\n{c['text']}" for c in selected_chunks])

    summary_result = None
    if gemini_client:
        prompt = f"""You are an academic tutor preparing a high-yield study sheet for university students.
Based on the following excerpts for subject '{selected_subject}', provide a key points summary.
Format your output STRICTLY as valid JSON:
{{
  "summary": "A 2 to 3 sentence comprehensive high-level summary of the core concepts.",
  "key_points": [
    "5 to 7 key bullet points highlighting important definitions, principles, or formulas"
  ]
}}

--- LECTURE EXCERPTS ---
{combined_text}
"""
        models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        for m in models_to_try:
            try:
                resp = gemini_client.models.generate_content(model=m, contents=prompt)
                raw = resp.text.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                parsed = json.loads(raw)
                if "key_points" in parsed and "summary" in parsed:
                    summary_result = parsed
                    break
            except Exception as e:
                print(f"Gemini summary generation failed: {e}")

    # Fallback extractive summary
    if not summary_result:
        all_sentences = []
        for c in selected_chunks:
            all_sentences.extend(split_into_sentences(c["text"]))
        # Pick sentences that have definition or key words
        key_sentences = [s for s in all_sentences if any(w in s.lower() for w in ['is', 'are', 'defined', 'used to', 'key', 'important', 'process', 'means'])]
        if len(key_sentences) < 5:
            key_sentences = all_sentences[:6]
        summary_result = {
            "summary": f"Key conceptual overview of {selected_subject} based on uploaded study materials.",
            "key_points": key_sentences[:6] if key_sentences else ["Review your uploaded lecture notes for comprehensive coverage."]
        }

    return jsonify({
        "subject": selected_subject,
        "summary": summary_result.get("summary", ""),
        "key_points": summary_result.get("key_points", []),
        "sources": sources
    })


@app.route("/quiz", methods=["POST"])
def generate_quiz():
    """
    Generates 5 multiple-choice questions from chunks of the selected subject.
    Returns: question, 4 options, correct answer, one-line explanation, source page.
    """
    data = request.get_json(silent=True) or request.form
    selected_subject = data.get("subject", "All").strip()

    if not CHUNKS:
        return jsonify({"error": "No documents uploaded. Please upload PDFs first."}), 400

    if selected_subject and selected_subject.lower() != "all":
        candidate_chunks = [c for c in CHUNKS if c["subject"].lower() == selected_subject.lower()]
        if not candidate_chunks:
            candidate_chunks = CHUNKS
            selected_subject = "All Notes"
    else:
        candidate_chunks = CHUNKS
        selected_subject = "All Notes"

    # Pick 5 random or distributed chunks
    sample_size = min(5, len(candidate_chunks))
    sample_chunks = random.sample(candidate_chunks, sample_size) if len(candidate_chunks) >= sample_size else candidate_chunks

    context_str = "\n\n".join([f"[Source: {c['file']} · p.{c['page']}]\n{c['text']}" for c in sample_chunks])

    quiz_questions = None
    if gemini_client:
        prompt = f"""You are an exam creator for university courses.
Generate exactly 5 multiple choice questions (MCQs) testing understanding of the subject '{selected_subject}' based strictly on the provided context.
Return ONLY valid JSON matching this schema:
[
  {{
    "question": "Clear, direct question text",
    "options": [
      "A) First realistic option",
      "B) Second realistic option",
      "C) Third realistic option",
      "D) Fourth realistic option"
    ],
    "correct_answer": "A",
    "explanation": "One clear line explaining why this answer is correct according to the notes.",
    "source": "Filename.pdf · p.X"
  }}
]
Ensure correct_answer is just the single letter: 'A', 'B', 'C', or 'D'.
Context:
{context_str}
"""
        models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        for m in models_to_try:
            try:
                resp = gemini_client.models.generate_content(model=m, contents=prompt)
                raw = resp.text.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                parsed = json.loads(raw)
                if isinstance(parsed, list) and len(parsed) >= 1:
                    quiz_questions = parsed[:5]
                    break
            except Exception as e:
                print(f"Gemini quiz generation failed: {e}")

    # Fallback Rule-Based Quiz Generator
    if not quiz_questions or len(quiz_questions) < 1:
        quiz_questions = []
        for i, chunk in enumerate(sample_chunks[:5]):
            sentences = split_into_sentences(chunk["text"])
            target_sentence = sentences[0] if sentences else chunk["text"][:100]
            words = [w for w in re.findall(r'\b[A-Za-z]{4,}\b', target_sentence) if w.lower() not in {'this', 'that', 'with', 'from', 'have', 'were', 'which'}]
            
            keyword = words[0] if words else "concept"
            masked_question = target_sentence.replace(keyword, "______", 1)
            if masked_question == target_sentence:
                masked_question = f"According to the notes, which statement is true regarding: {target_sentence[:60]}...?"
            else:
                masked_question = f"Fill in the blank: {masked_question}"
                
            quiz_questions.append({
                "question": masked_question,
                "options": [
                    f"A) {keyword}",
                    "B) Alternative hypothesis",
                    "C) Secondary parameter",
                    "D) Inverted index"
                ],
                "correct_answer": "A",
                "explanation": f"Stated directly in {chunk['file']} on page {chunk['page']}.",
                "source": f"{chunk['file']} · p.{chunk['page']}"
            })

    return jsonify({
        "subject": selected_subject,
        "questions": quiz_questions
    })


@app.route("/clear", methods=["POST"])
def clear_all():
    """Utility endpoint to reset in-memory document state."""
    global CHUNKS
    CHUNKS = []
    return jsonify({"success": True, "message": "All documents cleared."})


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Campus Q&A Bot at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
