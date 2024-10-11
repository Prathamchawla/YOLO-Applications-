import cv2
import streamlit as st
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('best.pt')

# Variables to store multiple ROI points
rois = []  # List to store each parking spot ROI
drawing = False  # True if the mouse is pressed
start_point = None  # Starting point for drawing the bounding box

# Mouse callback function to draw ROI boxes
def draw_roi(event, x, y, flags, param):
    global drawing, start_point, rois

    # Start drawing the box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    # Update the box while the mouse is moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_frame = param.copy()
            cv2.rectangle(temp_frame, start_point, (x, y), (0, 255, 0), 2)  # Green color for drawing
            cv2.imshow("Draw ROI", temp_frame)

    # Finish drawing and save the ROI when the mouse is released
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        rois.append((start_point, end_point))  # Save the ROI (start, end)
        cv2.rectangle(param, start_point, end_point, (0, 255, 0), 2)  # Green color for initial display
        cv2.imshow("Draw ROI", param)

def detect_parking_spaces(video_file):
    st.info("Running YOLO for Parking Space Detection...")

    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    # Get the first frame to draw multiple ROIs
    ret, first_frame = cap.read()
    if ret:
        # Resize the frame to 640x640 for easier drawing
        first_frame_resized = cv2.resize(first_frame, (640, 640))

        # Show the window to draw multiple ROIs
        cv2.namedWindow("Draw ROI")
        cv2.setMouseCallback("Draw ROI", draw_roi, first_frame_resized)

        while True:
            cv2.imshow("Draw ROI", first_frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    # Process the video and detect parking spaces
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 640x640 for consistency with ROI drawing
        frame_resized = cv2.resize(frame, (640, 640))
        frame_count += 1

        # YOLO detection (detect cars and trucks, class IDs: car=2, truck=7)
        results = model.predict(frame_resized,iou=0.25)
        detections = results[0].boxes.data
        detections = detections.detach().cpu().numpy()

        # Draw the saved parking ROIs (green by default)
        for idx, roi in enumerate(rois):
            cv2.rectangle(frame_resized, roi[0], roi[1], (0, 255, 0), 2)  # Green ROI by default

        # Check if detected vehicles (cars or trucks) are inside any parking ROI
        for index, row in pd.DataFrame(detections).iterrows():
            x1, y1, x2, y2, _, class_id = map(int, row[:6])

            # Only check for cars (class_id=2) and trucks (class_id=7)
            if class_id in [0]:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Check if the vehicle is inside any parking ROI
                for idx, roi in enumerate(rois):
                    if is_inside_roi(cx, cy, roi):
                        # Change ROI color to red if a car or truck is detected in it
                        cv2.rectangle(frame_resized, roi[0], roi[1], (0, 0, 255), 2)  # Red ROI for detected vehicles

                        # Draw a blue bounding box for the car or truck
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bbox for detected car/truck

        # Update Streamlit display with resized 640x640 frame
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, caption="Processing Parking Space Detection...", use_column_width=True)

    cap.release()

# Check if a point (x, y) is inside the defined ROI
def is_inside_roi(x, y, roi):
    """Check if the center point is inside the ROI."""
    x1, y1 = roi[0]
    x2, y2 = roi[1]
    return x1 < x < x2 and y1 < y < y2
