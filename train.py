from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from cnn_model import CNN
from data_loader import CIFAR10Dataset


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)

  
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Using device:", device)

  
    full_train = CIFAR10Dataset("data/train.npz")
    test_dataset = CIFAR10Dataset("data/test.npz")

    train_dataset, val_dataset = random_split(
        full_train,
        [45000, 5000],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    best_val_accuracy = -1.0
    best_epoch = 0
    history = []

    Path("models").mkdir(exist_ok=True)
    checkpoint_path = Path("models/cifar10_cnn.pth")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_images = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            
            optimizer.zero_grad()          
            outputs = model(images)        
            loss = criterion(outputs, labels)  
            loss.backward()                
            optimizer.step()               

            total_loss += loss.item() * labels.size(0)
            total_images += labels.size(0)

        average_loss = total_loss / total_images
        val_accuracy = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": average_loss,
            "validation_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {average_loss:.4f} | "
            f"Validation accuracy: {val_accuracy:.2%}"
        )

        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            cpu_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            torch.save(cpu_weights, checkpoint_path)

    weights = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(weights)

    test_accuracy = evaluate(model, test_loader, device)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Model saved to: {checkpoint_path}")

    results = {
        "seed": 42,
        "epochs": epochs,
        "batch_size": 128,
        "learning_rate": 0.001,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "test_accuracy": test_accuracy,
        "history": history,
    }

    Path("models/results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()