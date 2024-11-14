
# Infant Vision Simulation with PyTorch

This project simulates infant vision characteristics such as visual acuity and color perception based on age. The simulation uses a custom PyTorch `DataLoader` to apply transformations that mimic the visual limitations and developmental trajectory of infants.

## Overview

This project explores two main aspects of infant vision development:
1. **Visual Acuity**: Blurring images to simulate an infant's visual acuity, which gradually improves with age from approximately 20/600 at birth to 20/20 by around three years of age.
2. **Color Perception**: Converting images to grayscale for newborns to simulate their limited color vision, then gradually transitioning to full color perception over the first few months.

The transformations are implemented in a custom `Dataset` class in PyTorch, allowing for flexible control of age-dependent image transformations.

## Key Features

- **Flexible Visual Acuity Transformation**: Simulates the gradual improvement in acuity by adjusting the blurring intensity based on age in months, mapping from 20/600 (newborn) to 20/20 (adult vision).
- **Age-Based Color Perception**: Newborns see in grayscale; over time, the vision shifts from grayscale to full color, reflecting the maturation of color perception.
- **Performance Benchmarking**: Tests the loading performance with and without transformations to evaluate computational overhead.

## Code Structure

### `dataset.py`
Contains the `ImageDataset` class, which:
- Loads images from a specified directory.
- Applies transformations based on the `age_in_months` parameter to simulate the properties of infant vision:
  - **Visual Acuity**: Uses Gaussian blur with a radius that decreases as `age_in_months` increases.
  - **Color Perception**: Converts images to grayscale if `age_in_months` is below 4 months, with a gradual grayscale-to-color blend for ages 4-12 months.

### `utils/plot_utils.py`
Contains helper functions to plot batches of images with or without transformations for comparison.

### `benchmark_data_loading.py`
A script to benchmark the time taken to load a set of images with and without the infant vision transformations applied. Results are plotted to visualize the effect of transformations.

## Usage

1. **Setting Up the Dataset**:
   - Place your images in a directory (e.g., `dataset/`).
   - Update `img_dir` in the code to point to this directory.

2. **Running the Simulation**:
   ```bash
   python benchmark_data_loading.py
   ```

   This will benchmark the loading time and display batches of images for visual confirmation of the transformations.

3. **Adjusting Parameters**:
   - `age_in_months`: Change this parameter in the `benchmark_dataloading.py` script to simulate different stages of visual development.

## References

This project is based on findings from two key studies on infant visual development:

1. **Skelton, A. E., Maule, J., & Franklin, A. (2017). Infant Color Perception: Insight into Perceptual Development**. _Trends in Cognitive Sciences, 21_(4), 283–293.  
   This paper provides a comprehensive analysis of how color perception develops in infants. It discusses how infants initially perceive limited colors, transitioning from grayscale to full color over time as their visual system matures. This study informs the color perception transformations in the project, where images for newborns are converted to grayscale, with gradual color transition introduced between 4 and 12 months of age.

2. **Vogelsang, L., Gilad-Gutnick, S., Ehrenberg, E., Yonas, A., Diamond, S., & Sinha, P. (2018). Potential Downside of High Initial Visual Acuity**. _Proceedings of the National Academy of Sciences, 115_(44), 11333–11338. [Link to Paper](https://doi.org/10.1073/pnas.1800901115).  
   This research explores the visual acuity development in infants, demonstrating that infants initially have very low acuity (around 20/600) which improves over time to reach adult-level acuity (20/20) by around 36 months. This study supports the blurring transformations in the project, where images are heavily blurred for newborns, with blur intensity decreasing progressively with age.

These studies provide insight into the perceptual limitations and developmental progression of infant vision, informing the transformations applied in this project.

## Example Plots

The project includes visualization of transformations applied at different ages to provide a clearer understanding of the visual changes infants experience.

---

