# Applied Generative AI - Assignment 2

Adds a trained CIFAR-10 image classifier to the previous Module 3 FastAPI word-embedding service. The exact required architecture is implemented in PyTorch. The trained checkpoint is included, so the instructor does not need to train or download CIFAR-10 to run inference.

## Run with Docker (recommended for grading)

From the repository root, with Docker Desktop running:

```bash
docker build -t columbia-assignment2 .
docker run --rm -p 8000:8000 columbia-assignment2
```

Open http://localhost:8000/docs. Expand `POST /classify-image`, click **Try it out**, choose a JPEG/PNG image, then click **Execute**. A fixed example from the CIFAR-10 test set is provided in `examples/test_image.png`; its true label is in `examples/test_image.json`.

The Docker build requires internet to install packages and the spaCy model. Once built, inference needs no external model service or dataset download. CPU inference supports machines without a GPU. If port 8000 is occupied, use `-p 8001:8000` and open port 8001 instead.

## API examples

```bash
curl -X POST http://localhost:8000/classify-image \
  -F 'file=@examples/test_image.png'

curl 'http://localhost:8000/embedding?word=hello'

curl -X POST http://localhost:8000/get-embedding \
  -H 'Content-Type: application/json' -d '{"word":"hello"}'
```

The classifier returns `class_id`, `class`, `confidence`, probabilities for all ten classes, and model metadata. `confidence` is an uncalibrated softmax score, not a guarantee of correctness. Invalid/empty images return 400, files above 10 MB return 413, unsupported formats return 415, and a missing upload returns 422. RGB conversion also supports grayscale and RGBA PNGs.

## Exact assignment architecture

| Layer | Output shape (BCHW, then batch/features) |
|---|---|
| RGB input, resized to 64 x 64 | B x 3 x 64 x 64 |
| Conv2d: 16 filters, 3 x 3, stride 1, padding 1 + ReLU | B x 16 x 64 x 64 |
| MaxPool2d: 2 x 2, stride 2 | B x 16 x 32 x 32 |
| Conv2d: 32 filters, 3 x 3, stride 1, padding 1 + ReLU | B x 32 x 32 x 32 |
| MaxPool2d: 2 x 2, stride 2 | B x 32 x 16 x 16 |
| Flatten | B x 8192 |
| Linear + ReLU | B x 100 |
| Linear (logits) | B x 10 |

There are 825,398 trainable parameters. CrossEntropyLoss receives raw logits; softmax is applied only to API output. There are no extra convolutional layers, BatchNorm, or Dropout in the submitted model.

## Training and evaluation

CIFAR-10 contains 50,000 training and 10,000 test images. The training split is deterministically divided into 45,000 training and 5,000 validation examples using seed 42. The highest validation-accuracy checkpoint is selected; test accuracy is measured after training. All images are resized from 32 x 32 to the required 64 x 64, converted to tensors, and normalized per channel with mean 0.5 and standard deviation 0.5. The API uses the same preprocessing.

**Measured result:** 65.00% accuracy on all 10,000 held-out test images (loss 1.0058). Validation selected epoch 5. Training and evaluation took about 95 seconds on this Mac using MPS, excluding dataset preparation.

Training uses Adam, learning rate 0.001, batch size 128, and 5 epochs. See `models/metrics.json` for measured results and `models/training_history.json` for per-epoch training and validation measurements. GPU backend differences may prevent bit-identical reruns despite the fixed seed.

To train locally using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
python train.py --epochs 5
uvicorn main:app --host 0.0.0.0 --port 8000
```

`train.py` selects Apple MPS, CUDA, or CPU based on availability. The API always loads CPU weights for portability. Training overwrites the included checkpoint; keep a copy if comparing runs.

### Optional faster dataset mirror

The original dataset server was slow during this run, so the same CIFAR-10 splits were obtained from the public [uoft-cs/cifar10 mirror](https://huggingface.co/datasets/uoft-cs/cifar10). Mirror parquet files are verified by SHA-256 before conversion; labels and split membership are preserved. To reproduce that data path:

```bash
pip install pyarrow==19.0.1
python prepare_mirror.py
python train.py --epochs 5
```

If mirror files are absent, the loader uses torchvision's standard CIFAR-10 downloader. Dataset files are excluded from Git and Docker; trained weights are included.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover required model dimensions and parameter count, trained inference, probability consistency, grayscale conversion, invalid/oversized/missing input, and regression checks for both previous embedding routes. They require the included checkpoint, sample image, and spaCy model.

## Files

- `main.py`: FastAPI endpoints and startup model loading.
- `helper_lib/`: reusable model, data loader, training, evaluation, and checkpoint helpers.
- `train.py`: reproducible training entry point.
- `models/cifar10_cnn.pt`: selected trained weights and metadata.
- `models/metrics.json`, `models/training_history.json`: actual evaluation results.
- `Assignment2_Theory_Answers.pdf`: six theory answers with calculations.
- `THEORY_ANSWERS.md`: editable theory answers.
- `Dockerfile`, `requirements.txt`: deployment.

## Limitations

This is a classroom classifier restricted to ten CIFAR-10 categories. It always chooses one of those categories, including for unrelated images; there is no unknown-class detector. CIFAR-10 test performance does not establish accuracy on high-resolution real-world photos. Uploaded images are processed in memory and are not intentionally saved by the application. This local demo has no authentication or production load testing.

## References and assistance

Based on the instructor's Module 4 CNN/FCNN notebooks and reusable helper-library activity. Architecture follows Assignment2.pdf rather than the notebook's EnhancedCNN. AI assistance was used for implementation, debugging, documentation, and explanation of the calculations; the student should review the submission and follow the course's disclosure policy.
