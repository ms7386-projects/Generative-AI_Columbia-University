from pathlib import Path
import torch
from .model import AssignmentCNN
from .data_loader import CLASSES


def save_model(model, path, epoch, validation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'classes': CLASSES, 'epoch': epoch, 'validation': validation,
                'input_size': [3, 64, 64]}, path)


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = AssignmentCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint
