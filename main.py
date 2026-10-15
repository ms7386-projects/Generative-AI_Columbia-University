from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import spacy
from pathlib import Path
from io import BytesIO

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import UploadFile, File

from cnn_model import CNN
from data_loader import image_transform, CLASSES

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

# API 啟動時，載入一次已訓練好的 CNN
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "cifar10_cnn.pth"
)

cnn = CNN()

weights = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

cnn.load_state_dict(weights)
cnn.eval()

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
@app.post("/classify-image")
def classify_image(file: UploadFile = File(...)):
    # 限制上傳大小，避免一次讀取過大的檔案
    max_bytes = 10 * 1024 * 1024
    contents = file.file.read(max_bytes + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail="Image file must be no larger than 10 MB.",
        )

    try:
        with Image.open(BytesIO(contents)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise HTTPException(
                    status_code=415,
                    detail="Please upload a JPEG or PNG image.",
                )

            if source.width * source.height > 20_000_000:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large.",
                )

            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = image_transform(image).unsqueeze(0)

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Cannot read this image.",
        )

    with torch.no_grad():
        outputs = cnn(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        class_id = probabilities.argmax(dim=1).item()

    return {
        "class_id": class_id,
        "class": CLASSES[class_id],
        "confidence": probabilities[0, class_id].item(),
    }