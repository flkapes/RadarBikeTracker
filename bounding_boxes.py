from collections import deque
from typing import Dict, List, Tuple
import numpy as np
from depth import DepthProcessing
from visualizer import Visualizer
from matching import RadarMatcher


class BoundingBoxTracker:
    """Tracks bounding boxes across frames and handles interpolation."""

    def __init__(self, max_missing_frames: int = 25):
        self.track_history: Dict[int, Dict] = {}
        self.max_missing_frames = max_missing_frames

    def update_tracks(
        self,
        current_detections: Dict[int, np.ndarray],
        frame_idx: int,
        depth_map: np.ndarray,
        radar_data: Dict,
        fudge_factor: int,
        frame_rate: int,
        depth_processor: 'DepthProcessing',
        frame_buffer: deque,
    ):
        current_track_ids = set(current_detections.keys())

        # Update track history for missing tracks
        for track_id in list(self.track_history.keys()):
            if track_id not in current_track_ids:
                self.track_history[track_id]['missing_frames'].append(
                    frame_idx)
                if len(
                        self.track_history[track_id]['missing_frames']) > self.max_missing_frames:
                    # Remove track if it doesn’t reappear
                    del self.track_history[track_id]
            else:
                if len(self.track_history[track_id]['missing_frames']) > 0:
                    missing_frames = self.track_history[track_id]['missing_frames']
                    if len(missing_frames) <= self.max_missing_frames:
                        last_known_frame, last_known_bbox = self.track_history[track_id]['frames'][-1]
                        current_bbox = current_detections[track_id]
                        num_missing = len(missing_frames)
                        for i, missing_frame in enumerate(missing_frames):
                            alpha = (i + 1) / (num_missing + 1)
                            interpolated_bbox = last_known_bbox * \
                                (1 - alpha) + current_bbox * alpha
                            frame_timestamp = fudge_factor + last_known_frame // frame_rate
                            avg_depth = depth_processor.get_average_depth(
                                depth_map, interpolated_bbox)
                            if avg_depth is None:
                                continue
                            matched_data = RadarMatcher().match_with_radar(
                                [{"bbox": interpolated_bbox, "track_id": track_id, "average_depth": avg_depth}],
                                radar_data,
                                frame_timestamp
                            )
                            if matched_data:
                                text = (
                                    f"ID: {matched_data[0]['track_id']} | "
                                    f"Speed: {matched_data[0]['speed']} m/s | "
                                    f"Dist: {matched_data[0]['distance']} m | "
                                    f"Depth: {matched_data[0]['est_depth']} m"
                                )
                                # Modify frame in frame_buffer
                                for idx, (buf_frame_idx, buf_frame) in enumerate(
                                        frame_buffer):
                                    if buf_frame_idx == missing_frame:
                                        Visualizer().draw_interpolated_bbox(
                                            buf_frame, interpolated_bbox, text
                                        )
                                        frame_buffer[idx] = (
                                            buf_frame_idx, buf_frame)
                                        break
                        self.track_history[track_id]['missing_frames'] = []
                # Update frames with current frame and bbox
                self.track_history[track_id]['frames'].append(
                    (frame_idx, current_detections[track_id]))

        # Update track history for new tracks
        for track_id, bbox in current_detections.items():
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'frames': [(frame_idx, bbox)],
                    'missing_frames': []
                }
