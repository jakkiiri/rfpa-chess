#!/usr/bin/python3

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

# Load the model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'yolov8n.pt')  
model = YOLO(model_path)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    # Display the frame with bounding boxes
    cv2.imshow("YOLO Detection", frame)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
