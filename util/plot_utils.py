import matplotlib.pyplot as plt
import math
import os


def save_images(images, batch_idx, age_in_months, save_dir="output_images", applied_transforms=None, dataset_name=None):
    """Helper function to plot images in a grid with age and optional transformations or dataset name."""
    os.makedirs(save_dir, exist_ok=True)
    
    num_images = len(images)
    cols = 6
    rows = math.ceil(num_images / cols)

    # Determine the title based on available parameters
    title = f"Batch {batch_idx} - Age: {age_in_months} months"
    if dataset_name:
        title += f" - Dataset: {dataset_name}"
    if applied_transforms:
        title += f"\nTransforms: {applied_transforms}"

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5+1))
    fig.suptitle(title, fontsize=12)
    plt.subplots_adjust(top=1)  # Adjusts the position

    for i in range(rows * cols):
        if i < num_images:
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
            img = images[i].permute(1, 2, 0).cpu()  # Move to (H, W, C) and detach from GPU if needed

            # Apply grayscale colormap if conditions match
            grayscale_condition = (dataset_name == "ImageDataset" if dataset_name else True) and age_in_months < 4
            ax.imshow(img, cmap="gray" if grayscale_condition else None)
            ax.axis('off')  # Remove axes
        else:
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
            ax.axis('off')  # Turn off empty subplots

    # Save the plot as an image file
    file_path = os.path.join(save_dir, f"batch_{batch_idx}_age_{age_in_months}.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)  # Close the plot to avoid displaying it
    print(f"Saved batch {batch_idx} at age {age_in_months} months to {file_path}")
