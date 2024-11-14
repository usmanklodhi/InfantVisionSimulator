from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageOps
import os
import random
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, age_in_months=0):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = [(file, 1) for file in os.listdir(img_dir)]  # Label set to 1 for all images
        self.age_in_months = age_in_months

    def __len__(self):
        return len(self.img_labels)
    
    def calculate_blur_radius(self):
        """Calculate blur radius based on age to simulate visual acuity."""
        max_blur = 15  # Max blur radius for very poor visual acuity (e.g., newborns)
        acuity_scale = min(max(self.age_in_months / 36, 0), 1)  # Scale between 0 and 1
        # Full blur at age 0, no blur at age 36 months
        return max_blur * (1 - acuity_scale)
    
    def calculate_color_blend_ratio(self):
        """Calculate the blend ratio for color perception based on age."""
        if self.age_in_months < 4: # Assume infants under 4 months see in grayscale
            return 0  # Full grayscale
        elif self.age_in_months < 12:  # Convert grayscale back to RGB for blending
            return (self.age_in_months - 4) / 8  # Gradual transition from 4 to 12 months
        else:
            return 1  # Full color

    def apply_age_based_transformations(self, image):
        # Mapping age in months to visual acuity (20/600 to 20/20) using a non-linear function
        blur_radius = self.calculate_blur_radius()
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))

        # Color Perception (Grayscale to Color) based on age
        color_blend_ratio = self.calculate_color_blend_ratio()
        if color_blend_ratio < 1:
            grayscale_image = ImageOps.grayscale(image).convert("RGB")
            image = Image.blend(grayscale_image, image, color_blend_ratio)

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