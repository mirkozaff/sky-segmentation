# Random Forest Algorithm for Image Segmentation

This repository contains an implementation of the Random Forest algorithm for image segmentation using Python. Random Forest is a versatile and powerful machine learning algorithm that can be applied to various tasks, including image segmentation.

## Requirements

To run the code in this repository, you need to have the following dependencies installed by using:

```bash
pip install -r requirements.txt
```

## Getting Started

To get started with this repository, follow these steps:

```bash
python rf-segmentation.py --input path/to/input/image.jpg
```

Replace `path/to/input/image.jpg` with the path to the input image you want to perform segmentation on.

## Usage

The main file in this repository is `rf-segmentation.py`, which contains the implementation of the Random Forest segmentation algorithm Python. The algorithm takes an input image extracs the features from the image based on the sky regions, trains a Random Forest Classifier for segmentation and generates a mask image that separates the foreground and background regions.