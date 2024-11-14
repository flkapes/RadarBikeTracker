from pathlib import Path
from typing import Optional, List

from ultralytics import YOLO


class YOLOTracker:
    """Encapsulates YOLO tracking logic"""

    def __init__(self, model_path: str, task: str = "detect"):
        self.model = YOLO(Path(model_path), task=task)
        self.results = None

    def run_inference(
        self,
        source: str,
        tracker_path: str,
        imgsz: int,
        confidence: float,
        classes: Optional[List[int]] = None,
    ):
        """Runs inference using the loaded model on a video.

        Parameters
        ----------
            source: str
                The video file path to run inference on
            tracker_path: str
                The filepath of the tracking .yaml config file.
            imgsz: int
                The length of the longest edge to which the frames should be resized to.
                Resizing to 640 from 1080p means 640x384.
            confidence: float
                Exclude detections if their confidence is below this #.
            classes: list
                Only useful when using models with many classes, and one desired class.
        """
        if not classes:
            self.results = self.model.track(
                source=source,
                stream=True,
                persist=True,
                imgsz=imgsz,
                conf=confidence,
                tracker=tracker_path,
            )
        else:
            self.results = self.model.track(
                source=source,
                stream=True,
                persist=True,
                imgsz=imgsz,
                conf=confidence,
                tracker=tracker_path,
                classes=classes,
            )


print(dir(YOLOTracker))
