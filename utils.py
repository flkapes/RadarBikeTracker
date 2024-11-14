import json
import os
import glob


def checkpoint_finder(directory="checkpoints/", type="YOLO"):
    """Searches a folder for checkpoint files for use in the models

    Parameters
    ----------
        directory: str
            The directory to search for required file type
        type: str
            Use 'YOLO' to search for .pt files, otherwise
            specify the file-extension such as '.onnx'
    Returns
    -------
        list
            A list of checkpoint files
    """
    if type.lower() in "yolo":
        return glob.glob(f"{directory}/*.pt")
    else:
        if "." in type:
            return glob.glob(f"{directory}/*{type}")
        else:
            return glob.glob(f"{directory}/*.{type}")
