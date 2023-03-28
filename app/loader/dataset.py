import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A


class DatasetAugPil(Dataset):
    def __init__(self, file_paths: list[str], transform: A.Compose) -> None:
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path = self.file_paths[idx]
        image_np = np.empty(0)

        # Load PIL image
        with Image.open(file_path) as image:
            # Convert PIL image to numpy array
            image_np = np.array(image)
        # Apply transformations
        augmented = self.transform(image=image_np)
        return augmented['image']


class DatasetAugPilLabel(Dataset):
    def __init__(self, file_paths: list[str], labels: list[str], transform: A.Compose) -> None:
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        image_np = np.empty(0)

        # Load PIL image
        with Image.open(file_path) as image:
            # Convert PIL image to numpy array
            image_np = np.array(image)
        # Apply transformations
        augmented = self.transform(image=image_np)
        return augmented['image'], label
