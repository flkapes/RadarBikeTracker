from typing import List, Dict, Optional
import numpy as np


class OcclusionHandler:
    """Handles occlusions between bounding boxes."""

    def remove_occlusions(
        self, bboxes: List[Dict], min_area: float = 10.0
    ) -> List[Dict]:
        """Removes occluding parts of bounding boxes from each other.

        Parameters
        ----------
        bboxes : List[Dict]
            List of bounding box dictionaries with keys 'bbox', 'track_id', 'average_depth', etc.
        min_area : float, optional
            Minimum area for a bounding box to keep, by default 10.0.

        Returns
        -------
        List[Dict]
            Updated list of bounding boxes after occlusion handling.
        """

        def calculate_intersection(bbox1: Dict,
                                   bbox2: Dict) -> Optional[np.ndarray]:
            x1, y1, w1, h1 = bbox1['bbox']
            x2, y2, w2, h2 = bbox2['bbox']

            ix_min = max(x1, x2)
            iy_min = max(y1, y2)
            ix_max = min(x1 + w1, x2 + w2)
            iy_max = min(y1 + h1, y2 + h2)

            if ix_min < ix_max and iy_min < iy_max:
                return np.array(
                    [ix_min, iy_min, ix_max - ix_min, iy_max - iy_min],
                    dtype=np.float32,
                )
            else:
                return None

        updated_bboxes = []

        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox['bbox']
            original_area = w * h
            new_w, new_h = w, h

            for j, other_bbox in enumerate(bboxes):
                if i != j:
                    intersection = calculate_intersection(bbox, other_bbox)
                    if intersection is not None:
                        ix, iy, iw, ih = intersection
                        if iw < new_w:
                            new_w -= iw
                        if ih < new_h:
                            new_h -= ih

            new_area = new_w * new_h

            if new_area / original_area >= 0.10:
                updated_bbox = bbox.copy()
                updated_bbox['bbox'] = np.array(
                    [x, y, new_w, new_h], dtype=np.float32)
                updated_bboxes.append(updated_bbox)

        return updated_bboxes
