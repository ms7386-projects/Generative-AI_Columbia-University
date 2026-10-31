from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


PROJECT_DIR = Path(__file__).resolve().parent


def get_mnist_loader(batch_size=128):
    """Download MNIST and create a shuffled training data loader."""

    transform = transforms.Compose([
        # Convert pixel values from [0, 255] to [0, 1].
        transforms.ToTensor(),

        # Map pixel values from [0, 1] to [-1, 1].
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.MNIST(
        root=str(PROJECT_DIR / "data"),
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


if __name__ == "__main__":
    loader = get_mnist_loader()
    images, labels = next(iter(loader))

    print("Training images:", len(loader.dataset))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Pixel range:", images.min().item(), images.max().item())

    # Convert images back to [0, 1] for display.
    preview = (images[:64] + 1) / 2

    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    preview_path = output_dir / "mnist_preview.png"
    save_image(preview, str(preview_path), nrow=8)

    print("Preview saved to:", preview_path)