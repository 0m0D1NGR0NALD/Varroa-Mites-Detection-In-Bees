# Importing required libraries
import torch
import cv2

# Function to run detection
def detectx (frame, model):
    """
    >>> This function takes frame and model
    """
  frame = [frame]
  print(f"[INFO] Detecting. . . ")
  results = model(frame)
  # Display results
  results.show()
  # Frame predictions 
  print(results.xyxyn[0]) # Tensor
  print(results.pandas().xyxyn[0]) # Pandas
  print(results.xyxyn[0][:, -1]) # Tensor
  print(results.xyxyn[0][:, :-1]) # Tensor
  # Extract class label and BBox coordinates
  label, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
  # Return label and respective coordinates
  return label, coordinates

# Plot the BBox and results
def plot_boxes(results, frame,classes):
    """
    >>> This function takes results, frame and classes
    >>> results: contains labels and coordinates predicted by model on the given frame
    >>> classes: contains the labels
    """
    labels, coordinates = results
    print(labels)
