"""Train on CIFAR-10; select checkpoint on validation, evaluate test only at end."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from helper_lib.data_loader import get_data_loaders
from helper_lib.model import AssignmentCNN
from helper_lib.trainer import train_epoch
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import save_model, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    train, val, test = get_data_loaders(batch_size=args.batch_size)
    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, best = [], -1.0
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(model, train, criterion, optimizer, device)
        validation = evaluate_model(model, val, criterion, device)
        row = {'epoch': epoch, 'train': training, 'validation': validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation['accuracy'] > best:
            best = validation['accuracy']
            save_model(model, 'models/cifar10_cnn.pt', epoch, validation)
        Path('models/training_history.json').write_text(json.dumps(history, indent=2))
    selected, checkpoint = load_model('models/cifar10_cnn.pt')
    result = evaluate_model(selected.to(device), test, criterion, device)
    metrics = {'dataset_source': 'uoft-cs/cifar10 mirror' if Path('data/train.npz').exists() else 'torchvision CIFAR10', 'seed': 42, 'device': str(device), 'epochs_run': args.epochs,
               'selected_epoch': checkpoint['epoch'], 'batch_size': args.batch_size,
               'learning_rate': 0.001, 'train_samples': 45000, 'validation_samples': 5000,
               'test': result, 'training_and_evaluation_seconds': time.monotonic()-start,
               'parameter_count': sum(p.numel() for p in model.parameters())}
    Path('models/metrics.json').write_text(json.dumps(metrics, indent=2))
    # Fixed first test example, not selected for a successful prediction.
    from helper_lib.data_loader import get_dataset
    raw = get_dataset('./data', train=False)
    image, label = raw[0]
    image.save('examples/test_image.png')
    Path('examples/test_image.json').write_text(json.dumps({'test_index': 0, 'label': label, 'class': raw.classes[label]}, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
