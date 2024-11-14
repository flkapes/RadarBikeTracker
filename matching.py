from typing import List, Dict, Any
import numpy as np
from occlusions import OcclusionHandler


class RadarMatcher:
    """Matches bounding boxes with radar data."""

    def match_with_radar(
        self,
        bounding_boxes: List[Dict],
        radar_data: Dict[str, Any],
        frame_timestamp: int,
    ) -> List[Dict]:
        """Matches bounding boxes with radar data for the current timestamp.

        Parameters
        ----------
        bounding_boxes : List[Dict]
            List of bounding box dictionaries with keys 'bbox', 'track_id', 'average_depth', etc.
        radar_data : Dict[str, Any]
            Radar data loaded from a JSON file.
        frame_timestamp : int
            The timestamp of the current frame.

        Returns
        -------
        List[Dict]
            List of matched data containing bounding boxes and corresponding radar information.
        """
        matched_data = []

        # Retrieve radar data for the current frame timestamp if available
        radar_frame_data = radar_data.get(str(int(frame_timestamp)), [])
        if not radar_frame_data or not radar_frame_data[-1]:
            print(
                f"No car entries in radar data for timestamp {frame_timestamp}")
            return matched_data  # Return empty if no data is available
        else:
            car_entries = radar_frame_data[-1] if isinstance(
                radar_frame_data[-1], list) else []

        if len(bounding_boxes) > 1:
            occlusion_handler = OcclusionHandler()
            bounding_boxes = occlusion_handler.remove_occlusions(
                bounding_boxes)

        # Sort bounding boxes by normalized depth
        bounding_boxes_sorted = sorted(
            bounding_boxes,
            key=lambda x: 1.0 / x["average_depth"]
            if x["average_depth"] < 1.0
            else x["average_depth"],
        )

        # Sort radar entries by distance
        car_entries = sorted(car_entries, key=lambda x: x[0])

        areas = [bbox['bbox'][-2] * bbox['bbox'][-1]
                 for bbox in bounding_boxes_sorted]
        used_boxes = set()

        # Match each car entry in radar data to the closest bounding box
        for car_entry in car_entries:
            radar_distance, radar_speed = car_entry
            best_match = None
            best_depth_diff = float("inf")
            best_area = 0

            for idx, box in enumerate(bounding_boxes_sorted):
                if box["track_id"] in used_boxes:
                    continue  # Skip already matched boxes

                avg_depth = box["average_depth"]
                est_depth = 1.0 / avg_depth if avg_depth < 1.0 else avg_depth
                depth_diff = abs(est_depth - radar_distance)

                if depth_diff < best_depth_diff and depth_diff < 38 and best_area < areas[idx]:
                    best_match = {
                        "bbox": box["bbox"],
                        "track_id": box["track_id"],
                        "speed": radar_speed,
                        "distance": radar_distance,
                        "est_depth": est_depth,
                        'area': areas[idx]
                    }
                    best_area = areas[idx]
                    best_depth_diff = depth_diff

            if best_match:
                matched_data.append(best_match)
                used_boxes.add(best_match["track_id"])

        return matched_data
