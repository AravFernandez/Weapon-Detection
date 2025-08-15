# Weapon and Knife Detection System

This project is a system for detecting firearms and knives in images and videos using computer vision.

## Overview

The primary goal of this project is to provide an automated system for identifying weapons in visual media. It leverages a deep learning model to perform real-time object detection, which can be applied to still images or video streams. The system is capable of identifying different types of weapons and highlighting them with bounding boxes.

## Key Features

- **Image and Video Analysis:** Can process both individual image files and video files.
- **YOLOv8 Powered:** Utilizes the YOLOv8 model for fast and accurate object detection.
- **Image Preprocessing:** Includes tools for image preprocessing using wavelet transforms (Haar, Symlet, Daubechies) to enhance features for detection.

## Technical Details

- The detection logic is implemented in Python using the `ultralytics` library for YOLOv8 and `OpenCV` for image and video manipulation.
- The `preprocessing-images.py` script uses the `PyWavelets` library to apply various wavelet transforms.
- The system uses a pre-trained model located in the `runs/detect/Normal_Compressed/weights/` directory.
