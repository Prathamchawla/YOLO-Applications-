import cv2
import streamlit as st
import pandas as pd
import time
from ultralytics import YOLO
from tracker import Tracker
import numpy as np

# Initialize YOLO model and tracker
model = YOLO('yolo11s.pt')
tracker = Tracker()

# Function to resize ROI points
def resize_roi_points(roi_points, original_size=(640, 640), target_size=(1020, 500)):
    resized_points = []
    x_scale = target_size[0] / original_size[0]
    y_scale = target_size[1] / original_size[1]
    
    for point in roi_points:
        resized_point = (int(point[0] * x_scale), int(point[1] * y_scale))
        resized_points.append(resized_point)
    
    return resized_points

def vehicle_analytics(video_file, roi_points):
    st.info("Running YOLO for Vehicle Analytics...")

    # Resize ROI points to match the 1020x500 frame size
    roi_points = resize_roi_points(roi_points, original_size=(640, 640), target_size=(1020, 500))

    # Variables for speed calculation and line positions
    red_line_y = 198
    blue_line_y = 268
    offset = 6

    down = {}
    up = {}
    counter_down = []
    counter_up = []
    speeds = {}
    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(video_file)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))

        # YOLO detection
        results = model.predict(frame)
        a = results[0].boxes.data
        a = a.detach().cpu().numpy()
        px = pd.DataFrame(a).astype("float")
        vehicle_list = []

        # Iterate over detected objects and filter cars, trucks, motorcycles
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            class_id = int(row[5])
            
            # Check if the class ID is for car, truck, or motorcycle
            if class_id in [2, 3, 7]:  # Class IDs for Car, Motorcycle, Truck
                vehicle_list.append([x1, y1, x2, y2])

                # Draw bounding box for the vehicle (same color for all)
                bbox_color = (0, 255, 0)  # Green for all vehicles by default
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

        # Update tracker and process vehicles
        bbox_id = tracker.update(vehicle_list)
        for bbox in bbox_id:
            x3, y3, x4, y4, vehicle_id = bbox
            cx = int((x3 + x4) // 2)
            cy = int((y3 + y4) // 2)

            # Detect vehicles crossing red and blue lines
            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                down[vehicle_id] = time.time()
            if vehicle_id in down:
                if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                    elapsed_time = time.time() - down[vehicle_id]
                    if counter_down.count(vehicle_id) == 0:
                        counter_down.append(vehicle_id)
                        distance = 10  # meters between lines
                        speed_mps = distance / elapsed_time
                        speed_kph = speed_mps * 3.6
                        speeds[vehicle_id] = int(speed_kph)  # Store speed for the vehicle

            # Handle upward direction
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                up[vehicle_id] = time.time()
            if vehicle_id in up:
                if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                    elapsed_time_up = time.time() - up[vehicle_id]
                    if counter_up.count(vehicle_id) == 0:
                        counter_up.append(vehicle_id)
                        distance = 3  # meters
                        speed_mps_up = distance / elapsed_time_up
                        speed_kph_up = speed_mps_up * 3.6
                        speeds[vehicle_id] = int(speed_kph_up)  # Store speed for the vehicle

            # Draw speed on the top of the vehicle once generated
        #     if vehicle_id in speeds:
        #         cv2.putText(frame, str(speeds[vehicle_id]) + ' Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # # Draw red and blue lines
        cv2.line(frame, (0, red_line_y), (frame.shape[1], red_line_y), (0, 0, 255), 2)
        cv2.line(frame, (0, blue_line_y), (frame.shape[1], blue_line_y), (255, 0, 0), 2)

        # Display the vehicle count on the frame
        cv2.rectangle(frame, (0, 0), (250, 90), (0, 255, 255), -1)
        cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Check if the car is in the truck lane (ROI)
        if len(roi_points) > 1:
            cv2.polylines(frame, [np.array(roi_points)], isClosed=True, color=(0, 255, 255), thickness=2)  # Red ROI for Truck Lane
            cv2.putText(frame, 'Truck Lane', (roi_points[0][0], roi_points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            for bbox in bbox_id:
                x3, y3, x4, y4, vehicle_id = bbox
                cx = int((x3 + x4) // 2)
                cy = int((y3 + y4) // 2)

                if is_inside_roi(cx, cy, roi_points):
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Red bounding box for cars in Truck Lane
                    cv2.putText(frame, 'Wrong Lane', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Update Streamlit image display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, caption="Processing Vehicle Analytics...", use_column_width=True)

    cap.release()

def is_inside_roi(x, y, roi_points):
    """Check if a point (x, y) is inside the defined ROI."""
    contour = np.array(roi_points)
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0
