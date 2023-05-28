# GrabCut with Random Forest Algorithm

This repository contains an implementation of the GrabCut combined with a Random Forest classifier for image segmentation. 

## Requirements

To run the code in this repository, you need to have the following dependencies installed by using:

```bash
pip install -r requirements.txt
```

## Getting Started

To get started with this repository, follow these steps:

```bash
python main.py --input path/to/input_dir --mask path/to/output_dir
```

Replace `path/to/input_dir` with the path to the folder containing the input images. Replace `path/to/output_dir` with the path to output folder.

## Usage

The main file in this repository is `main.py`, which contains the implementation of the segmentation algorithm. The algorithm takes an input folder and generates a mask images that separates the foreground and background regions of sky pixels.