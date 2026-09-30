from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import spacy

app = FastAPI(
    title="SpaCy Word Embedding API",
    description="Module 3 Homework: FastAPI with SpaCy Word Embeddings"
)

# spaCy 
try:
    nlp = spacy.load("en_core_web_md")
except Exception:
    import spacy.cli
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

class VectorRequest(BaseModel):
    word: str

@app.get("/")
def home():
    return {"status": "online", "message": "Word Embedding API is running."}

# method 1
@app.post("/get-embedding")
def get_embedding_post(payload: VectorRequest):
    word_text = payload.word.strip()
    if not word_text:
        raise HTTPException(status_code=400, detail="Word cannot be empty.")
    
    doc = nlp(word_text)
    
    if not doc.has_vector or doc.vector_norm == 0:
        raise HTTPException(status_code=404, detail=f"No vector found for word: '{word_text}'")
    
    return {
        "word": word_text,
        "vector_dim": int(doc.vector.shape[0]),
        "vector": doc.vector.tolist()
    }

# method 2
@app.get("/embedding")
def get_embedding_get(word: str = Query(..., description="The word to get vector for")):
    word_text = word.strip()
    doc = nlp(word_text)
    
    if not doc.has_vector or doc.vector_norm == 0:
        raise HTTPException(status_code=404, detail=f"No vector found for '{word_text}'")
        
    return {
        "word": word_text,
        "vector_dim": int(doc.vector.shape[0]),
        "vector": doc.vector.tolist()
    }