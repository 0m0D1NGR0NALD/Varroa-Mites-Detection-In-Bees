# Importing required libraries
import torch
import cv2

# Function to run detection
def detectx (frame, model):
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
  
  labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
  
  return labels, cordinates
