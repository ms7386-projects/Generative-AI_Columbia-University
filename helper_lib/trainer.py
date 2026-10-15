import time


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    start = time.monotonic()
    for step, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.detach().argmax(1) == labels).sum().item()
        total += labels.size(0)
        if step % 100 == 0:
            print(f'  batch {step}/{len(loader)} loss={loss_sum/total:.4f} accuracy={correct/total:.4f}', flush=True)
    return {'loss': loss_sum / total, 'accuracy': correct / total,
            'samples': total, 'seconds': time.monotonic() - start}
