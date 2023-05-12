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
                # Labels
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            else:
                # BBox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                # Labels
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    return frame

# Main function
def main(image_path=None, video_path=None, video_out = None):
    print(f"[INFO] Loading model... ")
    # Loading the custom trained model
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')
    # Class names in string format
    classes = model.names
    
    if image_path != None:
        print(f"[INFO] Working with image: {image_path}")
        # Loading image from path and reading image
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detecting varroa mites in input image  
        results = detectx(frame, model = model)  
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Drawing customized bounding boxes
        frame = plot_boxes(results, frame, classes = classes)
        # Creating a window to display the result
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        # Loop to display image
        while True:
            cv2.imshow("Image", frame)
            # Set up "esc" button as trigger to exit loop and save output
            if cv2.waitKey(50000) & 0xFF == 27:
                print(f"[INFO] Exiting. . . ")
                # Save he output result
                cv2.imwrite("final_output.jpg", frame)
                break
    elif video_path !=None:
        print(f"[INFO] Working with video: {video_path}")
        # Read the video
        vid = cv2.VideoCapture(video_path)
        # Create the video writer if video output path is given
        if video_out:
            # Default VideoCapture returns float instead of integer
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_out, codec, fps, (width, height))
        frame_no = 1
        # Creating a window to display the result
        cv2.namedWindow("Output Video", cv2.WINDOW_NORMAL)
        # Loop to display video frames
        while True:
            ret, frame = vid.read()      
            if ret:
                print(f"[INFO] Working with frame {frame_no} ")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Detecting varroa mites in each frame of the video 
                results = detectx(frame, model = model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
