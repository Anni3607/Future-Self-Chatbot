
# futureself_core.py
# Modular core for FutureSelf project
import json, uuid, datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import os

DATA_DIR = Path("futureself_data")
DATA_DIR.mkdir(exist_ok=True)
MEMORY_FILE = DATA_DIR / "memory.json"
PERSONA_FILE = DATA_DIR / "persona.json"

def load_json(path, default):
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        return default

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# Initialize memory structure if absent
memory = load_json(MEMORY_FILE, {"conversations": [], "journal": [], "facts": [], "embeddings": []})
persona = load_json(PERSONA_FILE, {})

# Models (lazy loaded)
_embed_model = None
_gen_tokenizer = None
_gen_model = None
_sentiment = None
_reranker = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model

def get_generator():
    global _gen_tokenizer, _gen_model
    if _gen_model is None:
        _gen_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        _gen_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return _gen_tokenizer, _gen_model

def get_sentiment_pipeline():
    global _sentiment
    if _sentiment is None:
        _sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _sentiment

def get_reranker():
    global _reranker
    if _reranker is None:
        # Cross-Encoder for re-ranking
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _reranker

# Memory helpers
def add_memory_text(text, meta=None):
    emb = get_embed_model().encode(text, convert_to_tensor=False).tolist()
    entry = {"id": str(uuid.uuid4()), "text": text, "embedding": emb, "meta": meta or {}, "timestamp": datetime.datetime.utcnow().isoformat()}
    memory["embeddings"].append(entry)
    save_json(MEMORY_FILE, memory)
    return entry

def retrieve_similar(text, top_k=6, min_score=0.25, rerank_top_k=5):
    model = get_embed_model()
    q_emb = model.encode(text, convert_to_tensor=True)
    corpus_tensor = np.array(corpus_embs, dtype=np.float32)
    if not corpus_embs:
        return []
    corpus_tensor = np.array(corpus_embs)
    scores = util.cos_sim(q_emb, corpus_tensor)[0].cpu().numpy()
    idxs = scores.argsort()[::-1]
    candidates = []
    for i in idxs[:top_k]:
        candidates.append({"score": float(scores[i]), "entry": memory["embeddings"][i]})
    # rerank if available
    try:
        reranker = get_reranker()
        pairs = [(text, c["entry"]["text"]) for c in candidates]
        rerank_scores = reranker.predict(pairs)
        # attach rerank score
        for c, r in zip(candidates, rerank_scores):
            c["rerank_score"] = float(r)
        # sort by rerank_score
        candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:rerank_top_k]
    except Exception:
        # fallback to original similarity order
        candidates = candidates[:rerank_top_k]
    # filter by min_score on original sim
    candidates = [c for c in candidates if c["score"] >= min_score]
    return candidates

# Sentiment
def detect_sentiment(text):
    try:
        s = get_sentiment_pipeline()(text)[0]
        return {"label": s["label"], "score": float(s["score"])}
    except Exception:
        tb = __import__("textblob").TextBlob(text).sentiment
        return {"label":"NEUTRAL", "score": 0.5, "polarity": tb.polarity}

# Summarize memory (extractive naive summarizer: top-K retrieval + join)
def summarize_memory_for_context(top_k=6):
    if not memory.get("embeddings"):
        return "No memory."
    # take the last top_k embeddings by timestamp
    sorted_entries = sorted(memory["embeddings"], key=lambda e: e.get("timestamp",""), reverse=True)[:top_k]
    texts = [e["text"] for e in sorted_entries]
    # optionally run a short summarization pipeline if available
    try:
        summarizer = pipeline("summarization")
        joined = " ".join(texts)
        summary = summarizer(joined, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
        return summary
    except Exception:
        # fallback: join bullets
        return "\\n".join(["- " + t for t in texts])

# Safe prompt builder
def craft_system_prompt(user_input, persona_obj, use_retrieval=True):
    persona_summary = f"Name: {persona_obj.get('name','User')}. Age: {persona_obj.get('age','unknown')}. Values: {', '.join(persona_obj.get('core_values',[]))}."
    retrieved_texts = []
    if use_retrieval:
        retrieved = retrieve_similar(user_input, top_k=8, rerank_top_k=5)
        for r in retrieved:
            retrieved_texts.append(f"- {r['entry']['text']} (sim={r.get('score',0):.2f}, rerank={r.get('rerank_score',0):.2f})")
    memory_summary = summarize_memory_for_context()
    sentiment = detect_sentiment(user_input)
    parts = [
        "You are FutureSelf — wise, empathetic, short and actionable.",
        persona_summary,
        "Memory summary: " + memory_summary,
        "Top retrieved memories:",
        "\\n".join(retrieved_texts) if retrieved_texts else "None",
        f"Detected emotion: {sentiment}",
        "Follow CBT-style empathy and provide 1 quick exercise + 3 concrete next steps and a 60-year-old perspective line."
    ]
    return "\\n\\n".join(parts)

# Generate reply (safer generator usage)
def generate_reply(user_input, persona_obj, max_new_tokens=200):
    prompt = craft_system_prompt(user_input, persona_obj) + "\\n\\nUser: " + user_input + "\\nFutureSelf:"
    try:
        tokenizer, gen_model = get_generator()
        input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
        out = gen_model.generate(input_ids, max_length=input_ids.shape[-1] + max_new_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_p=0.9, temperature=0.7, top_k=50)
        reply = tokenizer.decode(out[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        # safety filter: strip dangerous requests
        if any(bad in reply.lower() for bad in ["kill", "bomb", "hack", "virus", "illegal"]):
            return "I can’t help with that. If you need help, please seek a safe or professional resource."
        return reply
    except Exception as e:
        # fallback templated reply if generation fails
        cb = "Try listing evidence for and against a worry. Then pick a tiny experiment to test it."
        return "I couldn't produce a long creative reply right now. Quick help: " + cb

# High-level respond wrapper
def respond_and_record(user_input, persona_obj, remember=True):
    if remember:
        add_memory_text(user_input, {"source":"user_input"})
    reply = generate_reply(user_input, persona_obj)
    # 60yr snippet
    p60 = {
        "advice": (persona_obj.get("short_bio","") and "Keep building habits; small consistent wins matter.") or "Keep curiosity alive and invest in relationships."
    }
    sim = []  # you can implement scenario simulator or import external module
    record = {"id": str(uuid.uuid4()), "user": user_input, "reply": reply, "p60": p60, "sim": sim, "ts": datetime.datetime.utcnow().isoformat()}
    memory["conversations"].append(record)
    save_json(MEMORY_FILE, memory)
    # save a human readable last response (for UI handshake)
    Path("futureself_last_response.json").write_text("Reply:\\n" + reply + "\\n\\n60yr snippet:\\n" + p60["advice"])
    return record
