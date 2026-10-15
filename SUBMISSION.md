# Assignment 2 Submission

Repository: https://github.com/ms7386-projects/Generative-AI_Columbia-University

## Included work

1. Exact specified CNN in `helper_lib/model.py`, trained on CIFAR-10.
2. Existing FastAPI embedding endpoints preserved; added `POST /classify-image`.
3. Docker deployment with trained weights included.
4. Six theory answers with working in `Assignment2_Theory_Answers.pdf`.

## Measured results

- Train/validation/test: 45,000 / 5,000 / 10,000 images.
- Five training epochs; epoch 5 selected by validation accuracy.
- Held-out test accuracy: **65.00%**; test cross-entropy loss: **1.0058**.
- Exact architecture: 825,398 trainable parameters.
- Automated tests: **8 passed**.
- Docker image built and started successfully on Apple Silicon.
- Docker HTTP inference verified using the fixed first CIFAR-10 test image: true label cat, predicted cat, softmax score about 0.531.

See README.md for build/run commands, local training instructions, data provenance, and limitations. No training is required to run the supplied API. The original course notebooks are references, not additional submission requirements.
