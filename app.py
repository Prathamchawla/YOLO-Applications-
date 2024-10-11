import cv2
import streamlit as st
import tempfile
import numpy as np
import person_loitering
import abandoned_object
import vehicle_analytics
import parking_space_detection  # Import the Parking Space Detection module

# Initialize session state
if 'roi_points' not in st.session_state:
    st.session_state['roi_points'] = []
if 'video_file' not in st.session_state:
    st.session_state['video_file'] = None
if 'prediction_ready' not in st.session_state:
    st.session_state['prediction_ready'] = False

def draw_roi(frame, roi_points):
    """Function to draw ROI using mouse in Streamlit for Person Loitering, Abandoned Object, and Vehicle Analytics."""
    drawing = {"value": False}
    
    def handle_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["value"] = True
            roi_points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drawing["value"]:
            roi_points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing["value"] = False

    # Create an empty image placeholder in Streamlit
    frame_placeholder = st.empty()

    cv2.namedWindow("Draw ROI")
    cv2.setMouseCallback("Draw ROI", handle_mouse)

    while True:
        if len(roi_points) > 1:
            cv2.polylines(frame, [np.array(roi_points)], isClosed=False, color=(0, 255, 0), thickness=2)

        # Display updated frame in Streamlit (with the same placeholder)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, caption="Draw ROI and press 'q' to finish", use_column_width=True)

        # Optionally also show in an OpenCV window (not necessary for Streamlit)
        cv2.imshow("Draw ROI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Application Title
st.title("Applications of YOLO")

# Sidebar Selection for YOLO Applications
application = st.sidebar.selectbox("Select YOLO Application", ["Person Loitering", "Abandoned Object", "Vehicle Analytics", "Parking Space Detection"])

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    st.session_state['video_file'] = tfile.name
    st.session_state['prediction_ready'] = False  # Reset prediction flag

    cap = cv2.VideoCapture(st.session_state['video_file'])
    ret, frame = cap.read()

    if ret:
        frame_resized = cv2.resize(frame, (640, 640))

        # Draw ROI only for applications that need it (Person Loitering, Abandoned Object, Vehicle Analytics)
        if application != "Parking Space Detection":
            if st.button("Draw ROI"):
                st.info("Click to draw the ROI on the image and press 'q' when done.")
                draw_roi(frame_resized, st.session_state['roi_points'])

                if st.session_state['roi_points']:
                    st.success("ROI defined.")

    # Only show the Predict button if video and ROI are loaded or if it's Parking Space Detection
    if st.session_state['video_file'] and (application == "Parking Space Detection" or st.session_state['roi_points']):
        if st.button("Predict"):
            st.session_state['prediction_ready'] = True

    # Start prediction based on the selected application
    if st.session_state['prediction_ready']:
        if application == "Person Loitering":
            person_loitering.person_loitering_detection(st.session_state['video_file'], st.session_state['roi_points'])
        elif application == "Abandoned Object":
            abandoned_object.abandoned_object_detection(st.session_state['video_file'], st.session_state['roi_points'])
        elif application == "Vehicle Analytics":
            vehicle_analytics.vehicle_analytics(st.session_state['video_file'], st.session_state['roi_points'])
        elif application == "Parking Space Detection":
            # For Parking Space Detection, we use the separate ROI drawing logic defined in the parking module
            parking_space_detection.detect_parking_spaces(st.session_state['video_file'])
