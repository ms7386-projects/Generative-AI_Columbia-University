"""Module 3 embeddings plus Assignment 2 image classification."""
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
import warnings

import spacy
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from helper_lib.data_loader import CLASSES, image_transform
from helper_lib.utils import load_model

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'cifar10_cnn.pt'
MAX_FILE_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(2)
    # Provision models during setup/build, never download inside an API request.
    app.state.nlp = spacy.load('en_core_web_md')
    app.state.cnn, app.state.checkpoint = load_model(MODEL_PATH)
    app.state.transform = image_transform()
    yield


app = FastAPI(title='Word Embedding and CIFAR-10 Classification API',
              description='Module 3 embeddings and Assignment 2 trained CNN.',
              lifespan=lifespan)


class VectorRequest(BaseModel):
    word: str


@app.get('/')
def home():
    return {'status': 'online', 'message': 'Word Embedding and Image Classification API is running.'}


def embedding(word: str):
    text = word.strip()
    if not text:
        raise HTTPException(400, 'Word cannot be empty.')
    doc = app.state.nlp(text)
    if not doc.has_vector or doc.vector_norm == 0:
        raise HTTPException(404, f'No vector found for word: {text!r}')
    return {'word': text, 'vector_dim': int(doc.vector.shape[0]), 'vector': doc.vector.tolist()}


@app.post('/get-embedding')
def get_embedding_post(payload: VectorRequest):
    return embedding(payload.word)


@app.get('/embedding')
def get_embedding_get(word: str = Query(..., description='The word to get vector for')):
    return embedding(word)


@app.post('/classify-image')
def classify_image(file: UploadFile = File(..., description='JPEG or PNG image; maximum 10 MB')):
    contents = file.file.read(MAX_FILE_BYTES + 1)
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(413, 'Image file exceeds 10 MB.')
    if not contents:
        raise HTTPException(400, 'Image file is empty.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(contents)) as source:
                if source.format not in {'JPEG', 'PNG'}:
                    raise HTTPException(415, 'Only JPEG and PNG images are supported.')
                if source.width * source.height > 20_000_000:
                    raise HTTPException(413, 'Image exceeds 20 million pixels.')
                image = ImageOps.exif_transpose(source).convert('RGB')
                tensor = app.state.transform(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(400, 'Invalid, corrupted, or excessively large image.')
    with torch.inference_mode():
        probabilities = app.state.cnn(tensor).softmax(dim=1)[0]
    index = int(probabilities.argmax().item())
    return {'class_id': index, 'class': CLASSES[index],
            'confidence': float(probabilities[index].item()),
            'probabilities': {name: float(probabilities[i].item()) for i, name in enumerate(CLASSES)},
            'model': 'AssignmentCNN',
            'note': 'Restricted to CIFAR-10 classes; confidence is an uncalibrated softmax score.'}
