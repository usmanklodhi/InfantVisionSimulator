from PIL import ImageFilter
import math

class VisualAcuityTransform:
    def __init__(self, age_in_months):
        """
        Simulates visual acuity based on the age of the infant.
        Args:
            age_in_months (int): Age of the infant in months.
        """
        self.age_in_months = age_in_months
        # Exponential decay for blur radius: rapid improvement initially, slower later
        self.blur_radius = 10 * math.exp(-age_in_months / 3)  # 20/600 -> 20/20

    def __call__(self, image):
        """
        Apply the transformation to the given image.
        Args:
            image (PIL.Image): Input image.
        Returns:
            PIL.Image: Blurred image to simulate visual acuity.
        """
        return image.filter(ImageFilter.GaussianBlur(self.blur_radius))
