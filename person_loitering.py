import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

def person_loitering_detection(video_file, roi_points):
    st.info("Running YOLO for Person Loitering Detection...")

    model = YOLO('yolo11s.pt')  # Load YOLO model
    output_video_path = "output_person_loitering.mp4"
    output_cap = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 640))

    cap = cv2.VideoCapture(video_file)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 640))

        if len(roi_points) > 1:
            cv2.polylines(frame_resized, [np.array(roi_points)], isClosed=False, color=(0, 255, 0), thickness=2)

        results = model(frame_resized, classes=[0])  # Detect persons

        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if is_inside_roi(center_x, center_y, roi_points):
                    label = "Person Loitering"
                    color = (0, 0, 255)
                else:
                    label = "Person"
                    color = (0, 255, 0)

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
