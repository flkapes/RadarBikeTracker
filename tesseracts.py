import cv2
import re
from doctr.io import DocumentFile
from doctr.models import kie_predictor

# Setup the KIE model outside the function to avoid reloading it each time
model = kie_predictor(
    det_arch="db_resnet50",
    reco_arch="parseq",
    pretrained=True)


def calculate_time_until_change(video_path):
    cap = cv2.VideoCapture(video_path)
    time_pattern = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
    date_pattern = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

    # Helper function to preprocess each frame
    def preprocess_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    initial_time = None
    frame_count = 1
    date = ""

    # Ensure the video opened correctly and framerate is available
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps:
        raise ValueError("Unable to retrieve framerate from the video.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess the entire frame
        preprocessed_frame = preprocess_image(frame)
        doc = DocumentFile.from_array(preprocessed_frame)[0]
        result = model(doc)

        # Locate the prediction that matches HH:mm:ss format
        time_prediction = next(
            (
                pred.value
                for pred in result.pages[0].predictions["words"]
                if time_pattern.match(pred.value)
            ),
            None,
        )
        if frame_count == 1:
            start_time = time_prediction
            date_prediction = next(
                (
                    pred.value
                    for pred in result.pages[0].predictions["words"]
                    if date_pattern.match(pred.value)
                ),
                None,
            )
        # Stop if the time changes from the initial observation
        if time_prediction:
            if initial_time is None:
                initial_time = time_prediction  # Set initial timestamp
            elif time_prediction != initial_time:
                break  # Stop if timestamp changes
            frame_count += 1

    cap.release()

    # Return elapsed time (frames divided by framerate)
    elapsed_time = float(frame_count) / 29.97
    print(f"{video_path} frame_count:", frame_count)
    return elapsed_time, date_prediction, start_time


def get_start_time(video_path):
    from datetime import datetime

    elapsed_time, date_str, time_str = calculate_time_until_change(video_path)
    combined_datetime = datetime.strptime(
        f"{date_str} {time_str}", "%d/%m/%Y %H:%M:%S")
    unix_timestamp = int(combined_datetime.timestamp())

    # Add 5 hours to the Unix timestamp
    adjusted_timestamp = float(unix_timestamp) + (3600.0) + (1 - elapsed_time)

    return unix_timestamp, round(adjusted_timestamp)
