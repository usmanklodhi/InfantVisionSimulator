from PIL import ImageFilter
import math


class VisualAcuityTransform:
    def __init__(self, age_in_months):
        self.age_in_months = age_in_months

        # Define the maximum blur radius (newborns) and minimum (perfect vision).
        max_blur = 10  # Corresponds to 20/600 vision
        min_blur = 0  # Corresponds to 20/20 vision

        # Update the decay formula to reflect 36 months timeline
        self.blur_radius = max(min_blur, max_blur * math.exp(-age_in_months / 12))  # 12-month time constant

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(self.blur_radius))
