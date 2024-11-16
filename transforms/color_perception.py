from PIL import ImageEnhance, Image
import numpy as np
import math

class ColorPerceptionTransform:
    def __init__(self, age_in_months):
        """
        Simulates color perception based on the age of the infant.
        Args:
            age_in_months (int): Age of the infant in months.
        """
        self.age_in_months = age_in_months

        self.red_green_sensitivity = 1.0 if age_in_months >= 3 else max(0.1, (age_in_months - 2) / 1)

        # Blue-yellow sensitivity (refined)
        self.blue_yellow_sensitivity = 0.0 if age_in_months < 4 else min(1.0, (age_in_months - 4) / 6)

        # Saturation adjustment (unchanged)
        self.saturation_factor = min(1.0, age_in_months / 12)


    def __call__(self, image):
        """
        Apply the transformation to the given image.
        Args:
            image (PIL.Image): Input image.
        Returns:
            PIL.Image: Image with adjusted color perception.
        """
        # Convert image to NumPy array
        np_image = np.array(image).astype(np.float32)

        # Calculate original brightness per pixel
        original_brightness = np_image.mean(axis=2, keepdims=True)

        # Apply transformations based on sensitivity
        if self.age_in_months <= 1:
            # Grayscale-like appearance
            luminance = 0.299 * np_image[:, :, 0] + 0.587 * np_image[:, :, 1] + 0.114 * np_image[:, :, 2]
            np_image[:, :, 0] = luminance  # No red-green sensitivity
            np_image[:, :, 1] = luminance  # No red-green sensitivity
            np_image[:, :, 2] = luminance  # No blue-yellow sensitivity
        else:
            # Apply red-green sensitivity
            np_image[:, :, 0] *= self.red_green_sensitivity  # Red channel
            np_image[:, :, 1] *= self.red_green_sensitivity  # Green channel

            # Apply blue-yellow sensitivity
            np_image[:, :, 2] *= self.blue_yellow_sensitivity  # Blue channel

        # Normalize brightness to match the original
        adjusted_brightness = np_image.mean(axis=2, keepdims=True)
        brightness_ratio = original_brightness / np.clip(adjusted_brightness, 1e-6, None)  # Avoid division by zero
        np_image *= brightness_ratio

        # Clip values to valid range
        np_image = np.clip(np_image, 0, 255)

        # Convert back to PIL Image
        adjusted_image = Image.fromarray(np_image.astype(np.uint8))

        # Enhance saturation for age-based perception
        enhancer = ImageEnhance.Color(adjusted_image)
        enhanced_image = enhancer.enhance(self.saturation_factor)

        return enhanced_image
