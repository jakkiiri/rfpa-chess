import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import glob

# Load the model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model1.pt')

model = YOLO(model_path)

# Path to folder containing images
input_folder = os.path.join(script_dir, 'input_images')
output_folder = os.path.join(script_dir, 'output_images')

os.makedirs(output_folder, exist_ok=True)

# Get list of image files
image_files = glob.glob(os.path.join(input_folder, '*.jpg')) + \
              glob.glob(os.path.join(input_folder, '*.png')) + \
              glob.glob(os.path.join(input_folder, '*.jpeg'))

if not image_files:
    print(f"No images found in {input_folder}")
    exit()

for image_path in image_files:
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to read {image_path}")
        continue

    # Run YOLO model inference
    results = model(frame)

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Extract box coordinates, confidence, and class
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Ensure it's a list of integers
                conf = float(box.conf[0])  # Convert to float
                cls = int(box.cls[0])  # Convert to integer

                # Get label name (if available)
                if cls < len(model.names):
                    label = f"{model.names[cls]}: {conf:.2f}"
                else:
                    label = f"Class {cls}: {conf:.2f}"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow("YOLO Detection", frame)
    cv2.waitKey(500)  # Display each image for 500 ms

    # Save the output image
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, frame)
    print(f"Processed {image_path} -> {output_path}")

# Clean up
cv2.destroyAllWindows()
