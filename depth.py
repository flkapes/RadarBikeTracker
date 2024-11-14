"""Contains all depth estimation logic"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL.Image import fromarray
from torch.nn.functional import interpolate
from transformers import pipeline


class DepthProcessing:
    """This class manages the use of Depth-Anything-V2 model."""

    def __init__(
            self,
            encoder: str,
            input_video_path: str,
            input_size: int = 518):
        self.config = self._get_config(encoder)
        self.input_size = input_size
        self.input_video_path = input_video_path
        self.file_name = Path(
            f"depth_maps/{encoder}_{input_size}_{Path(input_video_path).stem}.npy"
        )
        self.depth_pipeline = pipeline(
            model=self.config["model_name"], task="depth-estimation"
        )

        if self.file_name.exists():
            try:
                self.depth_map = np.load(str(self.file_name))
            except OSError as e:
                print(
                    f"File cannot be read or does not exist. Double check the file. {e}"
                )
        else:
            self.get_depth_map()

    def get_depth_map(self) -> None:
        """Initializes, and runs inference on the depth estimation model of choice and normalizes the output.

        Returns
        -------
            None
        """
        depth_maps = []
        input_video = cv2.VideoCapture(str(self.input_video_path))
        while input_video.isOpened():
            status, frame = input_video.read()
            if not status:
                break
            image = fromarray(frame)
            depth_map = self.depth_pipeline(image)["predicted_depth"]
            depth_map = interpolate(
                depth_map.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).numpy()
            normalized_depth = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min()
            )
            depth_maps.append(normalized_depth)
        input_video.release()
        self.depth_map = depth_maps
        self.save_depth_map(depth_maps)

    def get_average_depth(
        self, depth_map: np.ndarray, bbox: np.ndarray
    ) -> Optional[float]:
        """Uses bounding box coordinates to calculate the average depth of an object

        Parameters
        ----------
            depth_map: np.ndarray
                A single depth map for a single frame
            bbox: np.ndarray
                An array with the following structure: x_center, y_center, width, height
        """
        x_center, y_center, width, height = bbox
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Ensure coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return np.mean(depth_map[y1:y2, x1:x2])

    @staticmethod
    def _get_config(encoder: str) -> dict:
        """Returns the requested config for a given model name.

        Choices are limited to small, base, and large models.
        Returns base if model name is invalid.
        """
        model_configs = {
            "small": {
                "model_name": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
            },
            "base": {
                "model_name": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
            },
            "large": {
                "model_name": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            },
        }

        return model_configs.get(encoder, "base")

    def save_depth_map(self, depth_map: np.ndarray) -> None:
        """Saves a depth map from memory to the disk for reuse"""
        try:
            np.save(self.file_name, depth_map)
        except Exception as e:
            print(
                f"An exception occured during file saving. Please see the file {self.file_name}.\n{e}"
            )
