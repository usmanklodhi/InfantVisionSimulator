from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageOps
import os
import random
import logging
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, age_in_months=0):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = [(file, 1) for file in os.listdir(img_dir) if file.endswith(('.jpg', '.jpeg', '.png'))]
        self.age_in_months = age_in_months

    def __len__(self):
        return len(self.img_labels)
    
    def calculate_blur_radius(self):
        """Calculate blur radius based on age to simulate visual acuity."""
        max_blur = 15  # Max blur radius for very poor visual acuity (e.g., newborns)
        acuity_scale = min(max(self.age_in_months / 12, 0), 1)  # Scale between 0 and 1
        blur_radius = max_blur * (1 - acuity_scale)
        return blur_radius
    
    def calculate_color_blend_ratio(self):
        """Calculate the blend ratio for color perception based on age."""
        # Gradual color sensitivity based on Skelton's findings
        if 0 <= self.age_in_months < 2:
            blend_ratio = 0  # Grayscale
        elif 2 <= self.age_in_months < 3:
            blend_ratio = 0.3  # Primarily grayscale with some red-green sensitivity
        elif 3 <= self.age_in_months < 6:
            blend_ratio = 0.6  # Partial color sensitivity, more saturation for red and green
        elif 6 <= self.age_in_months < 12:
            blend_ratio = 0.8  # Broader color range with emphasis on blues and yellows
        else:
            blend_ratio = 1  # Full color perception
        return blend_ratio
    
    def apply_red_green_sensitivity(self, image, blend_ratio):
        """Apply red-green sensitivity for ages 2-6 months."""
        grayscale_image = ImageOps.grayscale(image).convert("RGB")
        red_green_filter = Image.new("RGB", image.size, (200, 100, 100))
        blended_image = Image.blend(grayscale_image, red_green_filter, blend_ratio)
        return Image.blend(blended_image, image, blend_ratio)
    
    def apply_blue_yellow_sensitivity(self, image, blend_ratio):
        """Add blue-yellow sensitivity for ages 6-12 months."""
        grayscale_image = ImageOps.grayscale(image).convert("RGB")
        blue_yellow_filter = Image.new("RGB", image.size, (200, 200, 150))
        blended_image = Image.blend(grayscale_image, blue_yellow_filter, blend_ratio)
        return Image.blend(blended_image, image, blend_ratio)

    def apply_age_based_transformations(self, image):
        # Mapping age in months to visual acuity (20/600 to 20/20) using a non-linear function
        blur_radius = self.calculate_blur_radius()
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))

        # Color Perception (Grayscale to Color) based on age
        color_blend_ratio = self.calculate_color_blend_ratio()
        if color_blend_ratio < 1:
            grayscale_image = ImageOps.grayscale(image).convert("RGB")
            if 2 <= self.age_in_months < 3:
                # Enhance reds and greens for ages 2-3 months
                image = self.apply_red_green_sensitivity(image, color_blend_ratio)
            elif 3 <= self.age_in_months < 6:
                # Increase red-green sensitivity with higher color saturation
                image = self.apply_red_green_sensitivity(image, color_blend_ratio)
            elif 6 <= self.age_in_months < 12:
                # Add sensitivity to blues and yellows in the transition phase
                image = self.apply_blue_yellow_sensitivity(image, color_blend_ratio)
            else:
                # Full color for ages 12 months and up
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