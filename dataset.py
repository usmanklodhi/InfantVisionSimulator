from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageOps
import os
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
        # Visual acuity ranges for different ages according to research
        if self.age_in_months < 2:
            blur_radius = 15  # Very low acuity (20/600 or worse)
        elif 2 <= self.age_in_months < 3:
            blur_radius = 12  # High blur for 20/400 acuity
        elif 3 <= self.age_in_months < 6:
            blur_radius = 8   # Moderate blur for 20/200 acuity
        elif 6 <= self.age_in_months < 12:
            blur_radius = 4   # Lower blur for 20/100 acuity
        else:
            blur_radius = 1   # Near adult-like acuity (20/20) at 12+ months
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
    
    def apply_color_sensitivity(self, image, blend_ratio, filter_color):
        """Apply color sensitivity based on age and specific color filter."""
        grayscale_image = ImageOps.grayscale(image).convert("RGB")
        color_filter = Image.new("RGB", image.size, filter_color)
        blended_image = Image.blend(grayscale_image, color_filter, blend_ratio)
        return Image.blend(blended_image, image, blend_ratio)

    def apply_age_based_transformations(self, image):
        # Mapping age in months to visual acuity (20/600 to 20/20) using a non-linear function
        blur_radius = self.calculate_blur_radius()
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Color Perception (Grayscale to Color) based on age
        color_blend_ratio = self.calculate_color_blend_ratio()

        # If full color, skip blending to save processing time
        if color_blend_ratio < 1:
            grayscale_image = ImageOps.grayscale(image).convert("RGB")
            if 2 <= self.age_in_months < 3:
                # Enhance reds and greens for ages 2-3 months
                image = self.apply_color_sensitivity(image, color_blend_ratio, (200, 100, 100))  # Red-green
            elif 3 <= self.age_in_months < 6:
                # Increase red-green sensitivity with higher color saturation
                image = self.apply_color_sensitivity(image, color_blend_ratio, (200, 100, 100))  # Echanced Red-green
            elif 6 <= self.age_in_months < 12:
                # Add sensitivity to blues and yellows in the transition phase
                image = self.apply_color_sensitivity(image, color_blend_ratio, (200, 200, 150))  # Blue-yellow
            else:
                # Full color for ages 12 months and up
                image = Image.blend(grayscale_image, image, color_blend_ratio)

        return image

    def __getitem__(self, idx):
        img_path, label = os.path.join(self.img_dir, self.img_labels[idx][0]), self.img_labels[idx][1]
        try:
            image = Image.open(img_path).convert("RGB")  # Ensure RGB mode for consistency
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return None, None

        # Apply age-based transformations
        image = self.apply_age_based_transformations(image)

        # Ensure final image shape consistency
        if self.transform:
            image = self.transform(image)
        if image.size(0) != 3:  # Checking channels after transformation to tensor
            logging.error(f"Inconsistent image channels for {img_path}")
            return None, None

        return image, label