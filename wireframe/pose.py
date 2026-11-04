"""2D and metric 3D body landmarks from MediaPipe's Pose Landmarker (33 joints)."""
import hashlib
from pathlib import Path
import urllib.request

import numpy as np

from .geometry import ROOT

JOINTS = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
          "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
          "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
          "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
          "left_foot_index", "right_foot_index"]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
         (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
         (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
         (29, 31), (30, 32), (27, 31), (28, 32)]
# Joints that can touch the board, and the board point that touches: (joint, roles it may use).
HAND_CONTACTS = {15: (0, 2, 3), 16: (0, 2, 3)}       # wrists: hand, start, finish holds
FOOT_CONTACTS = {31: (1, 0, 2), 32: (1, 0, 2)}       # toes: foot holds, or any hold used as a foot

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
MODEL_PATH = ROOT / "data/cache/models/pose_landmarker_heavy.task"
# The file this prototype was tested with. The URL serves "latest", so a newer model warns.
MODEL_SHA256 = "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"


def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        print(f"Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != MODEL_SHA256:
        print(f"Warning: {path.name} differs from the tested model ({digest})", flush=True)
    return path, digest


class MediaPipePose:
    """Runs the landmarker in video mode, so it tracks the climber between samples."""

    def __init__(self, model_path=None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        self.model_path, self.model_sha256 = ensure_model(model_path or MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4, min_tracking_confidence=0.4)
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def __call__(self, frame_rgb, timestamp_ms):
        height, width = frame_rgb.shape[:2]
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self.landmarker.detect_for_video(image, int(timestamp_ms))
        if not result.pose_landmarks:
            return None
        landmarks, world = result.pose_landmarks[0], result.pose_world_landmarks[0]
        return {
            "pixels": np.array([[p.x * width, p.y * height] for p in landmarks]),
            "visibility": np.array([min(p.visibility or 0, p.presence or 1) for p in landmarks]),
            "world": np.array([[p.x, p.y, p.z] for p in world]),
        }

    def close(self):
        self.landmarker.close()
