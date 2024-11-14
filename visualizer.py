from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict
import cv2
import numpy as np


class Visualizer:
    """Draws bounding boxes and associated information on frames."""

    def draw_detections(self, frame: np.ndarray, matched_data: List[Dict]):
        """Draws bounding boxes and texts on the frame.

        Parameters
        ----------
        frame : np.ndarray
            The video frame on which to draw.
        matched_data : List[Dict]
            List of matched data containing bounding boxes and associated information.
        """
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for data in matched_data:
            bbox = data["bbox"]
            x_center, y_center, width, height = bbox
            x1, y1 = int(round(x_center - width / 2)
                         ), int(round(y_center - height / 2))
            x2, y2 = int(round(x_center + width / 2)
                         ), int(round(y_center + height / 2))

            # Draw the bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Prepare text
            text = (
                f"ID: {data['track_id']} | Speed: {data['speed']} m/s | "
                f"Dist: {data['distance']} m | Depth: {data['est_depth']} m"
            )
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width, text_height = (
                text_bbox[2] - text_bbox[0],
                text_bbox[3] - text_bbox[1],
            )

            # Draw text background
            background_rect = [
                x1,
                y1 - text_height - 5,
                x1 + text_width + 5,
                y1]
            draw.rectangle(background_rect, fill="black")
            draw.text((x1, y1 - text_height - 5),
                      text, fill="white", font=font)

        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def draw_interpolated_bbox(
            self,
            frame: np.ndarray,
            bbox: np.ndarray,
            text: str):
        """Draws interpolated bounding boxes and text on a frame.

        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on.
        bbox : np.ndarray
            The interpolated bounding box coordinates.
        text : str
            The text to display with the bounding box.
        """
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        x_center, y_center, width, height = bbox
        x1 = int(round(x_center - width / 2))
        y1 = int(round(y_center - height / 2))
        x2 = int(round(x_center + width / 2))
        y2 = int(round(y_center + height / 2))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw text background
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = (
            text_bbox[2] - text_bbox[0],
            text_bbox[3] - text_bbox[1],
        )
        background_rect = [x1, y1 - text_height - 5, x1 + text_width + 5, y1]
        draw.rectangle(background_rect, fill="black")
        draw.text((x1, y1 - text_height - 5), text, fill="white", font=font)

        # Update the frame
        updated_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame[:, :] = updated_frame
