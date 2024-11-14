from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageOps
import os
import random
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, age_in_months=0):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = [(file, 1 if "cat" in file else 0) for file in os.listdir(img_dir)]
        self.age_in_months = age_in_months

    def __len__(self):
        return len(self.img_labels)

    def apply_age_based_transformations(self, image):
        # Mapping age in months to visual acuity (20/600 to 20/20) using a non-linear function
        max_blur = 15  # Max blur radius for very poor visual acuity
        acuity_scale = np.clip((self.age_in_months / 36) ** 2, 0, 1)  # Non-linear scaling
        blur_radius = max_blur * (1 - acuity_scale)  # Full blur at age 0, no blur at age 36 months
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))

        # Color Perception (Grayscale to Color) based on age
        if self.age_in_months < 4:  # Assume infants under 4 months see in grayscale
            image = ImageOps.grayscale(image)  # Convert to true grayscale (1-channel)
        elif self.age_in_months < 12:  # Transition phase (gradual color)
            grayscale = ImageOps.grayscale(image).convert("RGB")  # Convert grayscale back to RGB for blending
            blend_ratio = (self.age_in_months - 4) / 8  # Transition from 4 to 12 months
            image = Image.blend(grayscale, image, blend_ratio)

        return image

    def __getitem__(self, idx):
        img_path, label = os.path.join(self.img_dir, self.img_labels[idx][0]), self.img_labels[idx][1]
        image = Image.open(img_path).convert("RGB")

        # Apply age-based transformations
        image = self.apply_age_based_transformations(image)

        # Apply any additional transforms
        if self.transform:
            image = self.transform(image)

        return image, label