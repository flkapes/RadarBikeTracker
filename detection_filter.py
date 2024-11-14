from typing import List, Tuple
import numpy as np


class DetectionFilter:
    """Filters detections based on criteria like field of view."""

    def filter_in_fov(
        self,
        boxes: np.ndarray,
        track_ids: List[int],
        frame_width: int,
        frame_height: int,
        fov_angle: float = 45.0,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Filters detections to include only those within the field of view (FOV).

        Parameters
        ----------
        boxes : np.ndarray
            Array of bounding boxes in the format [x_center, y_center, width, height].
        track_ids : List[int]
            List of track IDs corresponding to the bounding boxes.
        frame_width : int
            Width of the video frame.
        frame_height : int
            Height of the video frame.
        fov_angle : float, optional
            Field of view angle in degrees, by default 45.0.

        Returns
        -------
        Tuple[List[np.ndarray], List[int]]
            Filtered bounding boxes and their corresponding track IDs.
        """
        fov_width = int(np.tan(np.radians(fov_angle / 2)) * frame_height)
        x_center_frame = frame_width // 2
        x_left = x_center_frame - fov_width // 2
        x_right = x_center_frame + fov_width // 2

        filtered_boxes = []
        filtered_ids = []

        for bbox, track_id in zip(boxes, track_ids):
            x_center_bbox, y_center_bbox, width_bbox, height_bbox = bbox
            x1 = x_center_bbox - width_bbox / 2
            x2 = x_center_bbox + width_bbox / 2

            if (x1 +
                width_bbox /
                2 >= x_left) and (x2 -
                                  width_bbox /
                                  2 <= x_right):
                filtered_boxes.append(bbox)
                filtered_ids.append(track_id)

        return filtered_boxes, filtered_ids
