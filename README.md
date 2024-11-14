# Garmin Bike Mounted Radar Car Tracker

Using video + radar data extracted from a Garmin rear-mounted camera, this program will attempt to match the correct tracks from the object detection model, with their relevant radar distances + speeds with the help of amazing monocular depth estimation models, and linear interpolation of bounding boxes to mitigate issues found during testing.

## Technologies Used
    1. Fine-tuned Ultralytics YOLO11 + BOTSort tracking
    2. Depth-Anything-V2
    3. OpenCV
    4. Django (for the webui)
    5. Celery (for Asyncronous tasks)

This is a project inspired by the research topic a friend is pursuing in University. 