import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

def abandoned_object_detection(video_file, roi_points):
    st.info("Running YOLO for Abandoned Object Detection...")

    model = YOLO('yolo11s.pt')
    output_video_path = "output_abandoned_object.mp4"
    output_cap = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 640))

    cap = cv2.VideoCapture(video_file)
    frame_placeholder = st.empty()
    object_timestamps = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 640))

        if len(roi_points) > 1:
            cv2.polylines(frame_resized, [np.array(roi_points)], isClosed=False, color=(0, 255, 0), thickness=2)

        results = model(frame_resized, classes=[0, 24, 28, 26])  # Detect person, handbag, suitcase, backpack

        person_boxes = []  # Store all detected person bounding boxes

        # First pass: Get all person detections
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                cls = int(cls)

                if cls == 0:  # Person class
                    person_boxes.append((x1, y1, x2, y2))
                    label = "Person"
                    color = (0, 255, 0)

                    # Draw bounding box for person
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Second pass: Process objects and detect abandonment
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                cls = int(cls)

                if cls in [24, 28, 26]:  # Handbag, Suitcase, Backpack classes
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    label = "Object"
                    color = (255, 0, 0)

                    # Check if object is inside the ROI
                    if is_inside_roi(center_x, center_y, roi_points):
                        object_id = f"{cls}_{center_x}_{center_y}"

                        # Check if object intersects with any person box
                        if not intersects_with_person(x1, y1, x2, y2, person_boxes):
                            # Check if the object has been stationary for 5 seconds
                            if object_id not in object_timestamps:
                                object_timestamps[object_id] = time.time()
                            else:
                                elapsed_time = time.time() - object_timestamps[object_id]
                                if elapsed_time >= 5:
                                    label = "Abandoned Object"
                                    color = (0, 0, 255)

                        # Draw bounding box for object
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Write the processed frame to the output video
        output_cap.write(frame_resized)

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_placeholder.image(frame_pil, caption="Processing video...", use_column_width=True)

    cap.release()
    output_cap.release()

    st.success("Prediction completed. Displaying the output video...")
    video_file = open(output_video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

def is_inside_roi(x, y, roi_points):
    contour = np.array(roi_points)
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

def intersects_with_person(x1, y1, x2, y2, person_boxes):
    """Check if the bounding box of an object intersects with any person box."""
    for (px1, py1, px2, py2) in person_boxes:
        # Check if the bounding boxes intersect
        if x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1:
            return True  # Intersects with a person
    return False  # No intersection
