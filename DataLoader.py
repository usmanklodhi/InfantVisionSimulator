import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from skimage import io
import numpy as np
import cv2  # OpenCV for image transformations (optional)

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, annotations, root_dir, transform=None, apply_blur=False, rgb_filter=None):
        self.annotations = pd.read_csv(annotations)
        self.root_dir = root_dir
        self.transform = transform
        self.apply_blur = apply_blur  # Optionally apply blur
        self.rgb_filter = rgb_filter  # Optionally apply RGB filter (e.g., adjust R, G, B intensities)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = io.imread(img_name)

        # Apply blur if specified
        if self.apply_blur:
            image = cv2.GaussianBlur(image, (7, 7), 0)  # Example kernel size for blurring

        # Apply RGB filter if specified
        if self.rgb_filter is not None:
            image = image * self.rgb_filter  # e.g., np.array([1, 0.8, 0.6]) to modify RGB channels

        landmarks = self.annotations.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
