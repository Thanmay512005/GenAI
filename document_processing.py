"""FAQ document parsing, validation, CRUD, and sentence-transformer embeddings."""
from __future__ import annotations
import csv, io, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_FAQ_PATH = Path(__file__).parent / "data" / "travel_faq.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
_encoder = None

@dataclass(frozen=True)
class FAQEntry:
    question: str
    answer: str
    category: str = "General"

    @property
    def content(self) -> str:
        return f"Question: {self.question}\nAnswer: {self.answer}"

def parse_faq_text(text: str) -> list[FAQEntry]:
    if not text or not text.strip():
        raise ValueError("FAQ text is empty.")
    entries=[]
    q=a=None
    for raw in text.splitlines():
        line=raw.strip()
        if not line: continue
        if re.match(r"^(Q|Question)\s*:", line, re.I):
            if q and a: entries.append(FAQEntry(q,a))
            q=line.split(":",1)[1].strip(); a=None
        elif re.match(r"^(A|Answer)\s*:", line, re.I) and q:
            a=line.split(":",1)[1].strip()
        elif a is not None:
            a += " " + line
    if q and a: entries.append(FAQEntry(q,a))
    if not entries: raise ValueError("No valid question/answer pairs found. Use Q: and A: lines.")
    return entries

def load_faq_entries(file_path=DEFAULT_FAQ_PATH) -> list[FAQEntry]:
    path=Path(file_path)
    if not path.exists(): return []
    return parse_faq_text(path.read_text(encoding="utf-8"))

def save_faq_entries(entries: Iterable[FAQEntry], file_path=DEFAULT_FAQ_PATH):
    path=Path(file_path); path.parent.mkdir(parents=True,exist_ok=True)
    chunks=[]
    for e in entries:
        chunks += [f"Q: {e.question.strip()}", f"A: {e.answer.strip()}", ""]
    path.write_text("\n".join(chunks).rstrip()+"\n",encoding="utf-8")

def add_single_faq_entry(question, answer, category="General", file_path=DEFAULT_FAQ_PATH):
    entries=load_faq_entries(file_path)
    entries.append(FAQEntry(question.strip(),answer.strip(),category.strip() or "General"))
    save_faq_entries(entries,file_path); return entries[-1]

def update_single_faq_entry(index, question, answer, category="General", file_path=DEFAULT_FAQ_PATH):
    entries=load_faq_entries(file_path)
    if index<0 or index>=len(entries): return False
    entries[index]=FAQEntry(question.strip(),answer.strip(),category.strip() or "General")
    save_faq_entries(entries,file_path); return True

def delete_single_faq_entry(index, file_path=DEFAULT_FAQ_PATH):
    entries=load_faq_entries(file_path)
    if index<0 or index>=len(entries): return False
    entries.pop(index); save_faq_entries(entries,file_path); return True

def parse_faq_csv(text):
    rows=list(csv.DictReader(io.StringIO(text)))
    result=[]
    for r in rows:
        q=(r.get("question") or r.get("Question") or "").strip()
        a=(r.get("answer") or r.get("Answer") or "").strip()
        if q and a: result.append(FAQEntry(q,a,(r.get("category") or "General").strip()))
    if not result: raise ValueError("CSV needs question and answer columns.")
    return result

def parse_faq_json(text):
    rows=json.loads(text); result=[]
    for r in rows:
        if r.get("question") and r.get("answer"):
            result.append(FAQEntry(r["question"].strip(),r["answer"].strip(),r.get("category","General")))
    if not result: raise ValueError("JSON needs question and answer fields.")
    return result

def parse_uploaded_faq(name, text):
    suffix=Path(name).suffix.lower()
    if suffix==".csv": return parse_faq_csv(text)
    if suffix==".json": return parse_faq_json(text)
    if suffix in (".txt",".md"): return parse_faq_text(text)
    raise ValueError("Upload a .txt, .md, .csv, or .json file.")

def export_faq_text(entries):
    return "\n\n".join(f"Q: {e.question}\nA: {e.answer}" for e in entries)+"\n"
def export_faq_csv(entries):
    s=io.StringIO(); w=csv.writer(s); w.writerow(["question","answer","category"])
    for e in entries: w.writerow([e.question,e.answer,e.category])
    return s.getvalue()
def export_faq_json(entries):
    return json.dumps([{"question":e.question,"answer":e.answer,"category":e.category} for e in entries],indent=2,ensure_ascii=False)

def get_embedding_model():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder=SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _encoder

def generate_embeddings(texts, model=None):
    if not texts: 
        import numpy as np
        return np.empty((0,EMBEDDING_DIMENSIONS),dtype="float32")
    encoder=model or get_embedding_model()
    vectors=encoder.encode(list(texts),convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    return vectors

def embed_faq_entries(entries, model=None):
    return generate_embeddings([e.content for e in entries],model=model)
