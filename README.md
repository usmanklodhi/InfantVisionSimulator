# Infant Vision Simulator

This repository contains the implementation of the Infant Vision Simulator, developed as part of a project for the Computational Visual Perception course. The project is based on two key research papers:

1. Potential Downside of High Initial Visual Acuity by Vogelsang.
2. Infant Color Perception: Insight into Perceptual Development by Skelton.

The simulator models infant vision by implementing image transformations that replicate the visual properties of infants as they age, such as:

-   Gradual improvement in visual acuity.
-   Age-dependent changes in color perception.

## Project Overview

### Objectives

The project is divided into five tasks:

1. Literature Search

Research and identify an infant vision property (beyond low initial acuity) and determine its scientific basis. This property is implemented as an image transformation.

2. Dataset/Dataloader Class Implementation

Develop a flexible `Dataset` and `Dataloader` that incorporate:
- Visual acuity transformation (parameterized by age in months, scaling from 20/600 to 20/20).

- The property identified in Task 1.

3. Testing the Dataloader

Verify that the dataloader works as expected by loading and visualizing transformed images.

4. Performance Evaluation

Compare the performance of the data loading process with and without the transformations using 100 images.

5. Report Writing

Prepare a two-page report highlighting:

-   The identified infant vision property and its implementation.
-   The evaluation results with supporting images and performance plots.

## Repository Structure

```
InfantVisionSimulator/
├── dataset/
│   ├── Test_image_6/         # Folder with 6 test images
│   ├── Test_image_100/       # Folder with 100 test images
├── src/
│   ├── dataset.py            # Dataset class for infant vision
│   ├── dataloader.py         # Dataloader implementation
├── transforms/
│   ├── visual_acuity.py      # Visual acuity transformation
│   ├── color_perception.py   # Color perception transformation
├── scripts/
│   ├── plot.py               # Script for visualizing transformations
│   ├── test_dataloader.py    # Script for testing the dataloader
├── output_images/            # Folder for storing output images
├── README.md                 # Project documentation
└── report.pdf                # Final report (to be added)


```

## Installation

### Prerequisites
Ensure you have the following installed:
- Python >= 3.8
- pip

### Setup
Clone the repository:
```
git clone https://github.com/<username>/InfantVisionSimulator.git
cd InfantVisionSimulator
```
Install the project as a Python package:
```
pip install -e .
```
This command will install the project in "editable mode," allowing to make changes to the source code without needing to reinstall the package.


## Usage

### 1. Running the Dataloader Test

To test the dataloader and visualize the image transformations:
```
python scripts/test_dataloader.py
```
### 2. Visualizing Transformed Images
To plot and save the transformed images for different ages:
```
python scripts/plot.py
```

## References

This project is based on findings from two key studies on infant visual development:

1. **Skelton, A. E., Maule, J., & Franklin, A. (2017). Infant Color Perception: Insight into Perceptual Development**. _Trends in Cognitive Sciences, 21_(4), 283–293.  
   This paper provides a comprehensive analysis of how color perception develops in infants. It discusses how infants initially perceive limited colors, transitioning from grayscale to full color over time as their visual system matures. This study informs the color perception transformations in the project, where images for newborns are converted to grayscale, with gradual color transition introduced between 4 and 12 months of age.

2. **Vogelsang, L., Gilad-Gutnick, S., Ehrenberg, E., Yonas, A., Diamond, S., & Sinha, P. (2018). Potential Downside of High Initial Visual Acuity**. _Proceedings of the National Academy of Sciences, 115_(44), 11333–11338. [Link to Paper](https://doi.org/10.1073/pnas.1800901115).  
   This research explores the visual acuity development in infants, demonstrating that infants initially have very low acuity (around 20/600) which improves over time to reach adult-level acuity (20/20) by around 36 months. This study supports the blurring transformations in the project, where images are heavily blurred for newborns, with blur intensity decreasing progressively with age.

These studies provide insight into the perceptual limitations and developmental progression of infant vision, informing the transformations applied in this project.



---
