import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DatasetPil(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with Image.open(file_path) as image:
            return image


class DatasetPilLabel(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        with Image.open(file_path) as image:
            return image, label


class DatasetAugPil(Dataset):
    def __init__(self, file_paths, transform):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image_np = np.empty(0)

        # Load PIL image
        with Image.open(file_path) as image:
            # Convert PIL image to numpy array
            image_np = np.array(image)
        # Apply transformations
        augmented = self.transform(image=image_np)
        # Convert numpy array back to PIL Image
        image_aug = Image.fromarray(augmented['image'])
        return image_aug


class DatasetAugPilLabel(Dataset):
    def __init__(self, file_paths, labels, transform):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        image_np = np.empty(0)

        # Load PIL image
        with Image.open(file_path) as image:
            # Convert PIL image to numpy array
            image_np = np.array(image)
        # Apply transformations
        augmented = self.transform(image=image_np)
        # Convert numpy array back to PIL Image
        image_aug = Image.fromarray(augmented['image'])
        return image_aug, label
