# GrabCut Algorithm with OpenCV and Python

This repository contains an implementation of the GrabCut algorithm using OpenCV and Python. GrabCut is an image segmentation technique that separates the foreground and background regions in an image.

## Requirements

To run the code in this repository, you need to have the following dependencies installed by using:

```bash
pip install -r requirements.txt
```

## Getting Started

To get started with this repository, follow these steps:

```bash
python grabcut.py --input path/to/input/image.jpg --mask path/to/output/mask.jpg
```

Replace `path/to/input/image.jpg` with the path to the input image you want to perform GrabCut on. Replace `path/to/output/mask.jpg` with the path to the input mask you want to perform GrabCut on.

## Usage

The main file in this repository is `grabcut.py`, which contains the implementation of the GrabCut algorithm using OpenCV and Python. The algorithm takes an input image and generates a mask image that separates the foreground and background regions.