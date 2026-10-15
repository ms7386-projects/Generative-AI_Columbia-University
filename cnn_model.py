import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        #ReLU → Pooling
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        #ReLU → Pooling
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 100 → ReLU → 10 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = CNN()
    example = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        output = model(example)

    print("Input shape:", example.shape)
    print("Output shape:", output.shape)