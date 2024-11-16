import time
from src.dataloader import create_dataloader
import matplotlib.pyplot as plt

def test_performance(data_dir, ages, num_images=100):
    """
    Test performance of the data loader with and without transformations for multiple ages.

    Args:
        data_dir (str): Path to the dataset directory.
        ages (list): List of ages in months for transformations.
        num_images (int): Number of images to load.
    """
    
    performance = {"ages": [], "without_transform": [], "with_transform": []}
    
    print(f"Testing performance for {num_images} images...")

    for age in ages:
        print(f"\n=== Testing for Age: {age} months ===")

        # Test without transformations
        print("\nWithout Transformations:")
        dataloader_no_transform = create_dataloader(data_dir, batch_size=1, age_in_months=age, use_transform=False)
        start_time = time.time()
        for i, (image, _) in enumerate(dataloader_no_transform):
            if i >= num_images - 1:
                break
        end_time = time.time()
        no_transform_time = end_time - start_time
        print(f"Time Taken (No Transform): {no_transform_time:.4f} seconds")

        # Test with transformations
        print("\nWith Transformations:")
        dataloader_with_transform = create_dataloader(data_dir, batch_size=1, age_in_months=age, use_transform=True)
        start_time = time.time()
        for i, (image, _) in enumerate(dataloader_with_transform):
            if i >= num_images - 1:
                break
        end_time = time.time()
        with_transform_time = end_time - start_time
        print(f"Time Taken (With Transform): {with_transform_time:.4f} seconds")

        # Performance summary for this age
        print(f"\nPerformance Summary for Age {age} months:")
        print(f"- Without Transform: {no_transform_time:.4f} seconds")
        print(f"- With Transform: {with_transform_time:.4f} seconds")
        
        # Store performance data
        performance["ages"].append(age)
        performance["without_transform"].append(no_transform_time)
        performance["with_transform"].append(with_transform_time)
        
    return performance

def plot_performance(performance):
    """
    Plot the performance comparison between with and without transformations.

    Args:
        performance (dict): Performance times with and without transformations.
    """
    ages = performance["ages"]
    without_transform = performance["without_transform"]
    with_transform = performance["with_transform"]

    plt.figure(figsize=(10, 6))
    plt.plot(ages, without_transform, label="Without Transform", marker='o')
    plt.plot(ages, with_transform, label="With Transform", marker='o')

    plt.title("Performance Comparison by Age (100 Images)", fontsize=14)
    plt.xlabel("Age (Months)", fontsize=12)
    plt.ylabel("Time (Seconds)", fontsize=12)
    plt.xticks(ages)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = "dataset/Test_image_100"  # Path to 100 test images
    ages = [0, 2, 3, 6, 12]  # Ages to simulate
    num_images = 100
    
    # Test performance and store results
    performance = test_performance(data_dir, ages, num_images=num_images)

    # Plot the performance results
    plot_performance(performance)