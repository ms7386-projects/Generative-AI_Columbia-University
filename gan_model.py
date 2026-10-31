import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generate 28x28 grayscale images from 100-dimensional noise."""

    def __init__(self):
        super().__init__()

        # Project each noise vector into a 128 x 7 x 7 feature map.
        self.fc = nn.Linear(100, 128 * 7 * 7)

        self.layers = nn.Sequential(
            # Upsample from 7x7 to 14x14.
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Upsample from 14x14 to a single-channel 28x28 image.
            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),

            # Constrain pixel values to the range [-1, 1].
            nn.Tanh(),
        )

    def forward(self, noise):
        x = self.fc(noise)
        x = x.reshape(noise.size(0), 128, 7, 7)
        return self.layers(x)


class Discriminator(nn.Module):
    """Estimate the probability that an input image is real."""

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # Downsample from 28x28 to 14x14.
            nn.Conv2d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),

            # Downsample from 14x14 to 7x7.
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Convert the feature maps into one real/fake probability.
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        return self.layers(images)


if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()

    # Use evaluation mode for the architecture check.
    generator.eval()
    discriminator.eval()

    # Check output shapes without computing gradients.
    with torch.no_grad():
        noise = torch.randn(2, 100)
        fake_images = generator(noise)
        predictions = discriminator(fake_images)

    assert fake_images.shape == (2, 1, 28, 28)
    assert predictions.shape == (2, 1)

    print("Noise shape:", noise.shape)
    print("Generated image shape:", fake_images.shape)
    print("Discriminator output shape:", predictions.shape)
    print("Architecture shape check passed.")