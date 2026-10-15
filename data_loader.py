import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# 訓練和 API 預測都要使用同樣的圖片處理方式
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])


class CIFAR10Dataset(Dataset):
    def __init__(self, file_path):
        with np.load(file_path, allow_pickle=False) as data:
            self.images = data["images"]
            self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        image = image_transform(image)
        label = int(self.labels[index])
        return image, label


if __name__ == "__main__":
    dataset = CIFAR10Dataset("data/train.npz")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    images, labels = next(iter(loader))

    print("Dataset size:", len(dataset))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("First image's class:", CLASSES[labels[0].item()])