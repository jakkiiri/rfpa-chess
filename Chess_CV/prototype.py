import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import copy

# ---------- 1. Helper Functions ----------
def get_chess_notation(grid_row, grid_col):
    """Convert grid position (0-7, 0-7) to chess notation (a1-h8)"""
    file = chr(ord('a') + grid_col)
    rank = grid_row + 1  # Adjusted to correctly map grid_row 0 to rank 1
    return f"{file}{rank}"

def detect_move(previous_state, current_state):
    """
    Compare two consecutive board states and return the move in algebraic notation.
    Returns None if no valid move is detected.
    """
    moved_pieces = []
    removed_pieces = []
    new_pieces = []

    # Find all changes
    for i in range(8):
        for j in range(8):
            prev = previous_state[i][j]
            curr = current_state[i][j]
            if prev != curr:
                if curr is None:
                    removed_pieces.append((i, j, prev))
                elif prev is None:
                    new_pieces.append((i, j, curr))
                else:
                    moved_pieces.append((i, j, prev, curr))

    # Handle normal moves
    if len(new_pieces) == 1 and len(removed_pieces) == 1:
        # Find source and destination
        dest_row, dest_col, piece = new_pieces[0]
        source_row, source_col, _ = removed_pieces[0]
        
        # Get chess notation
        source = get_chess_notation(source_row, source_col)
        dest = get_chess_notation(dest_row, dest_col)
        
        # Determine piece type (uppercase for white)
        piece_type = piece.upper() if piece.islower() else piece
        
        # Handle captures
        capture = current_state[dest_row][dest_col] != previous_state[dest_row][dest_col]
        
        # Build algebraic notation
        if piece_type == 'P':  # Pawn move
            if capture:
                move = f"{source[0]}x{dest}"
            else:
                move = dest
        else:  # Other pieces
            if capture:
                move = f"{piece_type}x{dest}"
            else:
                move = f"{piece_type}{dest}"
        
        # Check for castling
        if piece_type == 'K' and abs(source_col - dest_col) == 2:
            if dest_col == 6:  # Kingside
                return "O-O"
            elif dest_col == 2:  # Queenside
                return "O-O-O"
        
        return move
    
    # Handle pawn promotion
    elif len(new_pieces) == 1 and len(removed_pieces) == 1:
        dest_row, dest_col, new_piece = new_pieces[0]
        source_row, source_col, old_piece = removed_pieces[0]
        
        if old_piece.lower() == 'p' and ((dest_row == 0 and new_piece.islower()) or 
                                       (dest_row == 7 and new_piece.isupper())):
            promotion = new_piece.upper()
            return f"{get_chess_notation(dest_row, dest_col)}={promotion}"
    
    return None  # No clear move detected

def order_points(pts):
    """Arrange 4 corner points in consistent order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def validate_contour(contour, min_area=5000):
    """Ensure contour is a valid quadrilateral (convex and above a minimum area)."""
    if contour is None or len(contour) != 4:
        return False
    if not cv2.isContourConvex(contour):
        return False
    area = cv2.contourArea(contour)
    return area >= min_area

def find_board_contour(gray):
    """Try to find a quadrilateral contour (the board) using Canny + Hough fallback."""
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Method 1: Direct contour detection
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and validate_contour(approx):
            return approx

    # Method 2: Hough line-based detection fallback
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        vertical, horizontal = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 20:   # near-horizontal
                horizontal.append((x1, y1, x2, y2))
            elif abs(angle) > 70: # near-vertical
                vertical.append((x1, y1, x2, y2))

        intersections = []
        for v in vertical[:10]:
            for h in horizontal[:10]:
                x1, y1, x2, y2 = v
                x3, y3, x4, y4 = h
                denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
                if denom == 0:
                    continue
                px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
                py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
                if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
                    intersections.append([px, py])

        if len(intersections) >= 4:
            hull = cv2.convexHull(np.array(intersections, dtype="float32"))
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if validate_contour(approx):
                return approx

    return None

def draw_outline(frame, pts):
    """Draw the outer board outline in green on the original feed."""
    if len(pts) == 4:
        cv2.polylines(frame, [pts.astype(int)], True, (0,255,0), 2)
    return frame

# ---------- 2. Main Script ----------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model1.pt')
    model = YOLO(model_path)

    cap = cv2.VideoCapture(1)  # or 1 if you have multiple cameras
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    BOARD_SIZE = 700
    BORDER_OFFSET = 5

    # Separate offsets for top/bottom/left/right
    TOP_OFFSET = 50
    BOTTOM_OFFSET = 50
    LEFT_OFFSET = 50
    RIGHT_OFFSET = 50

    GRID_SIZE = 8

    previous_state = None
    print("Press SPACE to capture and detect, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        # 1) Find board contour
        contour = find_board_contour(gray)
        if contour is not None: 
            pts = contour.reshape(4, 2)
            frame = draw_outline(frame, pts)

        cv2.imshow("Chessboard Alignment", frame)
        key = cv2.waitKey(1) & 0xFFf
        if key == ord('q'):
            break

        if key == ord(' '):
            if contour is None or not validate_contour(contour):
                print("Board detection failed. Adjust camera or lighting.")
                continue

            try:
                # 2) Warp the board to BOARD_SIZE x BOARD_SIZE
                ordered = order_points(contour.reshape(4, 2))
                src_points = ordered.astype(np.float32)
                dst_points = np.array([
                    [BORDER_OFFSET, BORDER_OFFSET],
                    [BOARD_SIZE - BORDER_OFFSET, BORDER_OFFSET],
                    [BOARD_SIZE - BORDER_OFFSET, BOARD_SIZE - BORDER_OFFSET],
                    [BORDER_OFFSET, BOARD_SIZE - BORDER_OFFSET]
                ], dtype=np.float32)

                M = cv2.getPerspectiveTransform(src_points, dst_points)
                warped = cv2.warpPerspective(frame, M, (BOARD_SIZE, BOARD_SIZE))

                # 3) Draw the 8x8 grid lines with different top/bottom/left/right offsets
                warped_grid = warped.copy()

                # Outer rectangle in green
                cv2.rectangle(
                    warped_grid,
                    (LEFT_OFFSET, TOP_OFFSET),
                    (BOARD_SIZE - RIGHT_OFFSET, BOARD_SIZE - BOTTOM_OFFSET),
                    (0,255,0), 2
                )

                # Compute the effective grid width & height
                grid_width = (BOARD_SIZE - LEFT_OFFSET - RIGHT_OFFSET)
                grid_height = (BOARD_SIZE - TOP_OFFSET - BOTTOM_OFFSET)

                # We assume the board is still 8x8 squares
                cell_width = grid_width / GRID_SIZE
                cell_height = grid_height / GRID_SIZE

                # Draw vertical lines (yellow)
                for i in range(1, GRID_SIZE):
                    x = int(LEFT_OFFSET + i * cell_width)
                    cv2.line(warped_grid, (x, TOP_OFFSET), (x, BOARD_SIZE - BOTTOM_OFFSET), (0,255,255), 1)

                # Draw horizontal lines (yellow)
                for i in range(1, GRID_SIZE):
                    y = int(TOP_OFFSET + i * cell_height)
                    cv2.line(warped_grid, (LEFT_OFFSET, y), (BOARD_SIZE - RIGHT_OFFSET, y), (0,255,255), 1)

                # 4) YOLO detection on the warped board
                results = model(warped)
                board_state = [[None]*GRID_SIZE for _ in range(GRID_SIZE)]

                for result in results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        if conf < 0.5:
                            continue

                        # bounding box center
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # skip if outside the "inner" region
                        if not (LEFT_OFFSET <= cx < BOARD_SIZE - RIGHT_OFFSET and
                                TOP_OFFSET <= cy < BOARD_SIZE - BOTTOM_OFFSET):
                            continue

                        # Convert center to grid coordinates
                        grid_x = int((cx - LEFT_OFFSET) // cell_width)
                        # Invert grid_y to align bottom of the image as rank 1
                        grid_y = 7 - int((cy - TOP_OFFSET) // cell_height)

                        if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                            label = model.names[cls_id]
                            board_state[grid_y][grid_x] = label

                            cv2.rectangle(warped_grid, (x1,y1), (x2,y2), (0,255,0), 2)
                            cv2.putText(warped_grid, label, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # 5) Show final results
                cv2.imshow("Warped + Grid + Pieces", warped_grid)

                print("\nChessboard State:")
                for row in board_state:
                    print(row)

                # Detect move if we have a previous state
                if previous_state is not None:
                    detected_move = detect_move(previous_state, board_state)
                    if detected_move:
                        print(f"\nDetected move: {detected_move}")
                    else:
                        print("\nNo clear move detected")

                # Update previous state
                previous_state = copy.deepcopy(board_state)

            except Exception as e:
                print("Error in perspective transform or detection:", e)

    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated.")