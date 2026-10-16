# Assignment 2: CNN Image Classification API

This project adds a CIFAR-10 image classifier to the existing FastAPI
word-embedding service.

## CNN Architecture

Input: RGB image resized to 64 × 64.

1. Conv2d: 3 input channels, 16 filters, kernel 3, stride 1, padding 1
2. ReLU
3. MaxPool2d: kernel 2, stride 2
4. Conv2d: 16 input channels, 32 filters, kernel 3, stride 1, padding 1
5. ReLU
6. MaxPool2d: kernel 2, stride 2
7. Flatten: 32 × 16 × 16 = 8,192 features
8. Linear: 8,192 → 100
9. ReLU
10. Linear: 100 → 10 class logits

## Training and Results

- Dataset: CIFAR-10
- Training / validation / test split: 45,000 / 5,000 / 10,000 images
- Preprocessing: resize to 64 × 64, convert to tensor, normalize
  with mean 0.5 and standard deviation 0.5 for each RGB channel
- Optimizer: Adam, learning rate 0.001
- Loss: CrossEntropyLoss
- Batch size: 128
- Epochs: 5
- Best checkpoint selected using validation accuracy
- Best validation accuracy: 65.68%
- Held-out test accuracy: 65.00%

The trained weights are included in models/cifar10_cnn.pth.
Detailed training results are in models/results.json.
Training data is excluded from the repository.

CIFAR-10 data was obtained from the public uoft-cs/cifar10
Hugging Face mirror and converted to NumPy files.
The local data/train.npz and data/test.npz files contain
"images" and "labels" arrays. These files are needed only for
retraining, not for API inference.

## Run with Docker

Run these commands from the project root with Docker running:

```bash
docker build -t genai-cnn-api .
docker run --rm -p 8001:8000 genai-cnn-api
```

Open http://localhost:8001/docs.

The build requires internet to install packages and the spaCy model.
The trained CNN is included, so no retraining is required.

## Test Image Classification

1. Expand POST /classify-image.
2. Click "Try it out".
3. Upload test_image.png.
4. Click "Execute".

Alternatively:

```bash
curl -X POST http://localhost:8001/classify-image \
  -F "file=@test_image.png"
```

Example response:

```json
{
  "class_id": 3,
  "class": "cat",
  "confidence": 0.5309
}
```

The example is the first image from the CIFAR-10 test set.
Its correct label is cat.

## Existing Word Embedding API

POST /get-embedding accepts:

```json
{
  "word": "hello"
}
```

GET /embedding?word=hello is also available.

## Verification and Limitations

Docker image classification returned HTTP 200 and predicted cat
for the provided test image. The existing POST /get-embedding
endpoint also returned HTTP 200 with a 300-dimensional vector.

The classifier supports only the ten CIFAR-10 classes.
It will still choose one of these classes for unrelated images.
Confidence is a softmax score, not a calibrated probability
of correctness. Test-set accuracy does not guarantee the same
performance on other real-world images.