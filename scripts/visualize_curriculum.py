import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from transforms.color_perception import ColorPerceptionTransform
from transforms.visual_acuity import VisualAcuityTransform

# Load Tiny ImageNet dataset
train_data = load_dataset("zh-plus/tiny-imagenet")['train']

# Select a sample image from the dataset
example_image = train_data[3]['image']  # Assuming the image is a PIL Image object

# Ages for the curriculum
ages = [3, 6, 9, 12]

# Transformation application functions
def apply_visual_acuity(image, age):
    acuity_transform = VisualAcuityTransform(age)
    return acuity_transform(image)

def apply_color_perception(image, age):
    color_transform = ColorPerceptionTransform(age)
    return color_transform(image)

def apply_combined_transform(image, age):
    image = apply_visual_acuity(image, age)
    image = apply_color_perception(image, age)
    return image

# Create a grid for visualization
fig, axes = plt.subplots(4, len(ages), figsize=(9, 7), gridspec_kw={'wspace': 0.0005, 'hspace': 0.05})
fig.suptitle("Visualization of Curriculum Learning", fontsize=16)

# Add row labels for transformations
row_labels = ["Original Image", "Visual Acuity |", "Color Perception |", "Combined |"]
for row, label in enumerate(row_labels):
    axes[row, 0].text(-0.5, 0.5, label, fontsize=12, ha='center', va='center', rotation=90, transform=axes[row, 0].transAxes)
    axes[row, 0].axis("off")

# Plot images for each transformation and age
for col, age in enumerate(ages):
    # Original image
    axes[0, col].imshow(example_image)
    axes[0, col].axis("off")
    
    # Visual Acuity Transformation
    acuity_image = apply_visual_acuity(example_image, age)
    axes[1, col].imshow(acuity_image)
    axes[1, col].axis("off")
    
    # Color Perception Transformation
    color_image = apply_color_perception(example_image, age)
    axes[2, col].imshow(color_image)
    axes[2, col].axis("off")

    # Combined Transformation
    combined_image = apply_combined_transform(example_image, age)
    axes[3, col].imshow(combined_image)
    axes[3, col].axis("off")

    # Age annotation
    axes[0, col].set_title(f"{age} Months", fontsize=12)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("curriculum_visualization.png")
plt.show()
