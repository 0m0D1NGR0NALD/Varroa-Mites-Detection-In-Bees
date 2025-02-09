# Varroa Mite Detection using YOLOv5.

This repository contains an implementation of Varroa Mite detection using YOLOv5, a state-of-the-art object detection algorithm. Varroa mites are parasitic mites that infest honeybee colonies, causing significant damage to the bees and their hives. The purpose of this project is to develop an accurate and efficient system for automatically detecting Varroa mites in images or video footage.

## Dataset
Download Dataset used via https://zenodo.org/record/4085044#.ZFib6nZBzrd

## Usage
**Dataset Preparation:** Anotate dataset to customize it for YOLOv5 using Roboflow : https://app.roboflow.com

**Notebook Preparation:** The file titled 'custom_data.yaml' contains the customized ulterations for the data paths as well as the classes under investigation.

**Model Training:** Use the notebook titled : 'Varroa Mites Detection.ipynb' to train the varroa mites detection model stored in the path >> yolov5/runs/train/yolov5s_results/weights/best.pt, post training.

**Inference:** Use 'detect_and_plot.py' to carry out inference on the model using only single image and video data, a single prediction at a time.

## Input Sample Image :

![sample_image](https://user-images.githubusercontent.com/97228745/236753140-e448b3e6-a19c-4204-80ad-5efc2bfa05bb.png)

## Output Sample Image :

![final_output](https://user-images.githubusercontent.com/97228745/236753160-e9fb5cc6-a8a6-49d1-9f39-f095b28ab582.jpg)
