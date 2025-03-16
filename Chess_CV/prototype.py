import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

def order_points(pts):
    """Arrange quadrilateral points in consistent order (TL, TR, BR, BL)"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest difference
    return rect

def hough_board_detection(gray):
    """Fallback detection using Hough lines to find chessboard boundaries"""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                           minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return None
    
    vertical = []
    horizontal = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        if abs(angle) < 30:  # Horizontal
            horizontal.append((x1, y1, x2, y2))
        elif abs(angle) > 60:  # Vertical
            vertical.append((x1, y1, x2, y2))
    
    if len(vertical) < 2 or len(horizontal) < 2:
        return None
    
    # Find line intersections
    intersections = []
    for v_line in vertical[:10]:  # Limit to 10 strongest vertical
        for h_line in horizontal[:10]:  # Limit to 10 strongest horizontal
            x1, y1, x2, y2 = v_line
            x3, y3, x4, y4 = h_line
            
            # Calculate intersection point
            denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
            if denom == 0:
                continue
            px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
            intersections.append([px, py])
    
    if len(intersections) < 4:
        return None
    
    # Find convex hull of intersections
    hull = cv2.convexHull(np.array(intersections, dtype="float32"))
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    return approx if len(approx) == 4 else None

# ----- Initialization -----
script_dir = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(script_dir, 'model1.pt'))
output_folder = os.path.join(script_dir, 'output_images')
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

# Chessboard configuration
PATTERN_SIZE = (7, 7)  # Inner corners for 8x8 board
GRID_SIZE = (8, 8)     # Actual board squares
chessboard_state = [[None for _ in range(GRID_SIZE[1])] for _ in range(GRID_SIZE[0])]
frame_count = 0

print("Press SPACE to capture/process, Q to quit")

# ----- Main Loop -----
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture error")
        break
    
    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    if key == ord(' '):
        print("\n--- Processing Frame ---")
        original = frame.copy()
        
        # ----- Preprocessing -----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # ----- Board Detection -----
        board_contour = None
        
        # Method 1: Contour detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 5000:
                board_contour = approx
                break
        
        # Method 2: Hough lines fallback
        if board_contour is None:
            print("Contour detection failed, trying Hough lines...")
            board_contour = hough_board_detection(gray)
        
        # Validate detection
        if board_contour is None or len(board_contour) != 4:
            print("Board detection failed. Adjust view and try again.")
            cv2.imshow("Detection Failed", frame)
            cv2.waitKey(1000)
            continue
        
        # ----- Perspective Correction -----
        try:
            pts = board_contour.reshape(4, 2).astype("float32")
            ordered_pts = order_points(pts)
            
            # Calculate transformation matrix
            (tl, tr, br, bl) = ordered_pts
            width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
            height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
            dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(ordered_pts, dst)
            warped = cv2.warpPerspective(original, M, (int(width), int(height)))
            
            # Visualize detection
            cv2.drawContours(frame, [board_contour], -1, (0,0,255), 3)
            cv2.imshow("Board Detection", frame)
        except Exception as e:
            print(f"Perspective error: {str(e)}")
            continue
        
        # ----- Chessboard Corner Detection -----
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_gray = cv2.GaussianBlur(warped_gray, (3,3), 0)
        ret, corners = cv2.findChessboardCorners(
            warped_gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  cv2.CALIB_CB_FAST_CHECK
        )
        
        if not ret:
            print("Corners not detected in warped image")
            continue
        
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(warped_gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(warped, PATTERN_SIZE, corners, ret)
        cv2.imshow("Warped with Corners", warped)
        
        # ----- Piece Detection & Mapping -----
        results = model(original)
        chessboard_state = [[None]*GRID_SIZE[1] for _ in range(GRID_SIZE[0])]
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                # Extract detection info
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if conf < 0.5:  # Confidence threshold
                    continue
                
                # Transform center point to warped space
                center = np.array([[(x1+x2)//2, (y1+y2)//2]], dtype="float32")
                warped_center = cv2.perspectiveTransform(center.reshape(1,1,2), M)[0][0]
                
                # Find nearest chessboard square
                min_dist = float('inf')
                grid_pos = (-1, -1)
                
                for i in range(PATTERN_SIZE[0]):
                    for j in range(PATTERN_SIZE[1]):
                        idx = i * PATTERN_SIZE[1] + j
                        corner_x, corner_y = corners[idx][0]
                        distance = np.hypot(warped_center[0]-corner_x, warped_center[1]-corner_y)
                        
                        if distance < min_dist:
                            min_dist = distance
                            grid_pos = (i, j)
                
                # Update board state (convert pattern to grid coordinates)
                if grid_pos != (-1, -1) and min_dist < 50:  # Distance threshold
                    row = grid_pos[0]
                    col = grid_pos[1]
                    chessboard_state[row][col] = model.names[cls_id]
                    
                    # Draw annotations
                    cv2.rectangle(original, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(original, f"{model.names[cls_id]} {conf:.2f}", 
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # ----- Save & Display Results -----
        output_path = os.path.join(output_folder, f"detection_{frame_count}.jpg")
        cv2.imwrite(output_path, original)
        frame_count += 1
        
        cv2.imshow("Final Detection", original)
        print("\nCurrent Board State:")
        for row in chessboard_state:
            print(row)
        print("----------------------")

# ----- Cleanup -----
cap.release()
cv2.destroyAllWindows()
print("Program terminated")