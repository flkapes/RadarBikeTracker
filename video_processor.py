import os
from collections import deque
from typing import Dict, Any, List
import cv2
import numpy as np
from detection_filter import DetectionFilter
from depth import DepthProcessing
from matching import RadarMatcher
from bounding_boxes import BoundingBoxTracker
from visualizer import Visualizer
from YOLO import YOLOTracker


class VideoProcessor:
    """Manages the overall video processing workflow."""

    def __init__(
        self,
        input_video_path: str,
        radar_data_path: str,
        output_dir: str,
        fudge_factor: int,
    ):
        self.input_video_path = input_video_path
        self.radar_data_path = radar_data_path
        self.output_dir = output_dir
        self.fudge_factor = fudge_factor

        self.input_video = cv2.VideoCapture(input_video_path)
        self.frame_rate = int(self.input_video.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.depth_processor = DepthProcessing(
            encoder="base", input_video_path=input_video_path
        )
        self.depth_maps = self.depth_processor.depth_map

        self.detection_filter = DetectionFilter()
        self.radar_matcher = RadarMatcher()
        self.bounding_box_tracker = BoundingBoxTracker()
        self.visualizer = Visualizer()

        self.output_frames = []
        self.max_missing_frames = 25
        self.frame_buffer = deque([], maxlen=self.max_missing_frames)

        # Load radar data
        self.radar_data = self.load_radar_data()

    def load_radar_data(self) -> Dict[str, Any]:
        """Loads radar data from a JSON file."""
        import json

        with open(self.radar_data_path, "r") as file:
            radar_data = json.load(file)
        return radar_data

    def process_video(self):
        """Processes the video frame by frame."""
        input_video = cv2.VideoCapture(self.input_video_path)
        yolo_tracker = YOLOTracker(model_path="your_model_path.pt")
        yolo_tracker.run_inference(
            source=self.input_video_path,
            tracker_path="botsort.yaml",
            imgsz=640,
            confidence=0.7,
        )
        results = yolo_tracker.results

        depth_history = {}
        ignored_ids = set()

        for frame_idx, (frame_result, depth_map) in enumerate(
            zip(results, self.depth_maps)
        ):
            ret, fresh_frame = input_video.read()
            if not ret:
                break

            frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_to_draw = fresh_frame.copy()

            if (
                frame_result is None
                or frame_result.boxes is None
                or len(frame_result.boxes) == 0
            ):
                self.output_frames.append(frame_to_draw)
                self.frame_buffer.append((frame_idx, frame_to_draw))
                if len(self.frame_buffer) == self.max_missing_frames:
                    old_frame_idx, old_frame = self.frame_buffer.popleft()
                    self.output_frames.append(old_frame)
                continue

            # Extract bounding boxes and track IDs
            boxes = (
                frame_result.boxes.xywh.cpu().numpy()
                if frame_result.boxes.xywh is not None
                else []
            )
            track_ids = (
                frame_result.boxes.id.int().cpu().tolist()
                if frame_result.boxes.id is not None
                else []
            )

            if boxes.size == 0 or len(track_ids) == 0:
                self.output_frames.append(frame_to_draw)
                self.frame_buffer.append((frame_idx, frame_to_draw))
                if len(self.frame_buffer) == self.max_missing_frames:
                    old_frame_idx, old_frame = self.frame_buffer.popleft()
                    self.output_frames.append(old_frame)
                continue

            # Filter detections
            filtered_boxes, filtered_ids = self.detection_filter.filter_in_fov(
                boxes, track_ids, frame_width, frame_height
            )
            bbox_dict = {
                track_id: bbox for bbox,
                track_id in zip(
                    filtered_boxes,
                    filtered_ids)}

            # Update tracks
            self.bounding_box_tracker.update_tracks(
                current_detections=bbox_dict,
                frame_idx=frame_idx,
                depth_map=depth_map,
                radar_data=self.radar_data,
                fudge_factor=self.fudge_factor,
                frame_rate=self.frame_rate,
                depth_processor=self.depth_processor,
                frame_buffer=self.frame_buffer,
            )

            # Prepare boxes with depth information
            boxes_with_depth = []
            for track_id, bbox in bbox_dict.items():
                avg_depth = self.depth_processor.get_average_depth(
                    depth_map, bbox)
                if avg_depth is not None:
                    boxes_with_depth.append(
                        {
                            "bbox": bbox,
                            "average_depth": avg_depth,
                            "track_id": track_id,
                        }
                    )

            # Match with radar data
            frame_timestamp = (int(input_video.get(
                cv2.CAP_PROP_POS_MSEC) / 1000) + self.fudge_factor)
            matched_data = self.radar_matcher.match_with_radar(
                boxes_with_depth, self.radar_data, frame_timestamp
            )

            # Draw detections
            if matched_data:
                frame_to_draw = self.visualizer.draw_detections(
                    frame_to_draw, matched_data)

            self.frame_buffer.append((frame_idx, frame_to_draw))

            if len(self.frame_buffer) == self.max_missing_frames:
                old_frame_idx, old_frame = self.frame_buffer.popleft()
                self.output_frames.append(old_frame)

        # Write remaining frames
        while self.frame_buffer:
            old_frame_idx, old_frame = self.frame_buffer.popleft()
            self.output_frames.append(old_frame)

        input_video.release()
        self.save_output_video()

    def save_output_video(self):
        """Writes the processed frames to an output video file."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        output_video_path = os.path.join(
            self.output_dir, "processed_video.mp4")
        height, width = self.output_frames[0].shape[:2]
        out = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.frame_rate,
            (width, height),
        )
        for frame in self.output_frames:
            out.write(frame)
        out.release()
        print(f"Processed video saved to {output_video_path}")
