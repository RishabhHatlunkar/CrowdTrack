import cv2
import numpy as np
from ultralytics import YOLO
from utils import detect_people, get_zone_id, draw_zone_grid

# -------------------- CONFIG --------------------
GRID_ROWS, GRID_COLS = 3, 3
LOW_THRESHOLD = 2
HIGH_THRESHOLD = 10
DETECTION_CONFIDENCE = 0.5  # Not used directly unless passed to detect_people

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Initialize webcam (use CAP_DSHOW for Windows compatibility)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Unable to access webcam.")
    exit()

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    # Detect people
    detections = detect_people(model, frame)  # If your function supports conf, use: conf=DETECTION_CONFIDENCE

    # Initialize zone count
    zone_counts = [0] * (GRID_ROWS * GRID_COLS)

    # Draw zone grid
    draw_zone_grid(frame, GRID_ROWS, GRID_COLS)

    # Count people in each zone
    for det in detections:
        x1, y1, x2, y2, conf = det
        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
        zone_id = get_zone_id(xc, yc, frame_w, frame_h, GRID_ROWS, GRID_COLS)
        zone_counts[zone_id] += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (xc, yc), 4, (0, 0, 255), -1)

    # Overlay zone counts with alerts
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            zone_id = i * GRID_COLS + j
            count = zone_counts[zone_id]
            x = j * frame_w // GRID_COLS + 10
            y = i * frame_h // GRID_ROWS + 25

            # Color and label based on count
            if count >= HIGH_THRESHOLD:
                color = (0, 0, 255)  # Red
                label = f"🔥 Zone {zone_id} : {count}"
            elif count >= LOW_THRESHOLD:
                color = (0, 255, 255)  # Yellow
                label = f"⚠️ Zone {zone_id} : {count}"
            else:
                color = (255, 255, 255)  # White
                label = f"Zone {zone_id} : {count}"

            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display frame
    cv2.imshow("🛡️ Crowd Monitor - YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()