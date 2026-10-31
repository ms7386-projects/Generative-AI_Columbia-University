import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

from gan_model import Generator, Discriminator
from gan_data import get_mnist_loader


PROJECT_DIR = Path(__file__).resolve().parent


def train():
    # Set a seed for repeatable initialization and sampling.
    torch.manual_seed(42)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    epochs = 10
    batch_size = 128
    learning_rate = 0.0002

    model_dir = PROJECT_DIR / "models"
    sample_dir = PROJECT_DIR / "outputs" / "gan_samples"
    model_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    loader = get_mnist_loader(batch_size=batch_size)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # The discriminator already applies Sigmoid.
    criterion = nn.BCELoss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    # Reuse the same noise to compare samples across epochs.
    fixed_noise = torch.randn(64, 100).to(device)
    history = []

    print(f"Using device: {device}", flush=True)

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0
        total_images = 0

        for step, (real_images, _) in enumerate(loader, start=1):
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            real_targets = torch.ones(
                current_batch_size, 1, device=device
            )
            fake_targets = torch.zeros(
                current_batch_size, 1, device=device
            )

            # Step 1: Train the discriminator on real and fake images.
            optimizer_d.zero_grad(set_to_none=True)

            noise = torch.randn(current_batch_size, 100).to(device)
            fake_images = generator(noise)

            real_predictions = discriminator(real_images)
            fake_predictions = discriminator(fake_images.detach())

            d_loss_real = criterion(real_predictions, real_targets)
            d_loss_fake = criterion(fake_predictions, fake_targets)
            d_loss = (d_loss_real + d_loss_fake) / 2

            d_loss.backward()
            optimizer_d.step()

            # Step 2: Train the generator to receive "real" predictions.
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)

            optimizer_g.zero_grad(set_to_none=True)

            predictions = discriminator(fake_images)
            g_loss = criterion(predictions, real_targets)

            g_loss.backward()
            optimizer_g.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            total_d_loss += d_loss.item() * current_batch_size
            total_g_loss += g_loss.item() * current_batch_size
            total_images += current_batch_size

            if step % 100 == 0 or step == len(loader):
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Batch {step}/{len(loader)} | "
                    f"D loss: {d_loss.item():.4f} | "
                    f"G loss: {g_loss.item():.4f}",
                    flush=True,
                )

        # Save generated samples without updating BatchNorm statistics.
        generator.eval()
        with torch.no_grad():
            samples = generator(fixed_noise).cpu()
            samples = ((samples + 1) / 2).clamp(0, 1)

        save_image(
            samples,
            str(sample_dir / f"epoch_{epoch:02d}.png"),
            nrow=8,
        )

        # Save CPU weights for portable API loading.
        generator_weights = {
            name: value.detach().cpu()
            for name, value in generator.state_dict().items()
        }
        torch.save(
            generator_weights,
            model_dir / "mnist_generator.pth",
        )

        epoch_result = {
            "epoch": epoch,
            "d_loss": total_d_loss / total_images,
            "g_loss": total_g_loss / total_images,
            "seconds": time.perf_counter() - start_time,
        }
        history.append(epoch_result)

        with open(model_dir / "gan_history.json", "w") as file:
            json.dump(history, file, indent=2)

        print(
            f"Epoch {epoch} complete | "
            f"Average D loss: {epoch_result['d_loss']:.4f} | "
            f"Average G loss: {epoch_result['g_loss']:.4f} | "
            f"Time: {epoch_result['seconds']:.1f}s",
            flush=True,
        )

    print("Training complete.")
    print("Generator saved to:", model_dir / "mnist_generator.pth")
    print("Generated samples saved to:", sample_dir)


if __name__ == "__main__":
    train()