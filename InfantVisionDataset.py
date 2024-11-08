import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class InfantVisionDataset(Dataset):
    def __init__(self, image_paths, stage='grayscale'):
        """
        Args:
            image_paths (list of str): List of paths to the images.
            stage (str): The stage of color perception to simulate.
                         Options are 'grayscale', 'red-green', 'full_color'.
        """
        self.image_paths = image_paths
        self.stage = stage
        self.transform = self.get_transform()

    def get_transform(self):
        # Define transformations based on the color perception stage
        if self.stage == 'grayscale':
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])
        elif self.stage == 'red-green':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(self.red_green_transform)
            ])
        elif self.stage == 'full_color':
            return transforms.ToTensor()
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

    def red_green_transform(self, img):
        # Reduce blue channel to simulate infant red-green color sensitivity
        img[:, :, 2] = 0  # Zero out blue channel
        return img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and apply appropriate transformation
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)
        return img, self.stage
