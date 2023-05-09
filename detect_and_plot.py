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
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    
    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")
    
    # Looping through the detections
    for i in range(n):
        row = coordinates[i]
        print(row[4])
        # We are discarding everything below this value
        if row[4] >= 0.5: # Threshold value for detection.
            print(f"[INFO] Extracting BBox coordinates. . . ")
            # BBOx coordniates
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            # Labels text
            text = classes[int(labels[i])]
            if text == 'varroa-mite':
                # BBox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
