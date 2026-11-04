"""Wireframe extraction: geometry identities and an end-to-end synthetic video with ground truth.

Needs the `wireframe` extra (`uv sync --extra wireframe`); skipped without OpenCV.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import cv2  # noqa: F401
except ImportError:  # pragma: no cover
    cv2 = None


@unittest.skipIf(cv2 is None, "needs the wireframe extra")
class GeometryTest(unittest.TestCase):
    def test_world_round_trip_and_tilt(self):
        from wireframe.geometry import board_to_world, world_to_board
        points = np.random.default_rng(0).normal(size=(10, 3))
        np.testing.assert_allclose(world_to_board(board_to_world(points, 40), 40), points, atol=1e-12)
        # The top of an overhanging board leans out into the room; its normal tips down.
        top = board_to_world([0, 3.0, 0], 40)
        self.assertGreater(top[2], 1.9)
        self.assertLess(board_to_world([0, 0, 1], 40)[1], 0)

    def test_camera_recovered_from_its_homography(self):
        from wireframe.geometry import board_to_world, camera_from_homography
        from wireframe.synthetic import look_at
        truth = look_at(np.array([1.4, 1.0, 5.0]), board_to_world([1.8, 1.9, 0], 35), 35, 1300, (1080, 1920))
        camera = camera_from_homography(truth["H"], (1080, 1920))
        self.assertEqual(camera["focal_source"], "homography")
        self.assertAlmostEqual(camera["focal"], 1300, delta=1)
        np.testing.assert_allclose(camera["center_board"], truth["center_board"], atol=1e-3)
        self.assertGreater(camera["center_board"][2], 0, "camera must be on the room side")

    def test_lift_exact_skeleton(self):
        from wireframe.geometry import board_to_world, camera_from_homography, lift_skeleton, project
        from wireframe.synthetic import body, look_at
        truth = look_at(np.array([1.6, 1.1, 5.2]), board_to_world([1.8, 1.9, 0.3], 40), 40, 1380, (1080, 1920))
        camera = camera_from_homography(truth["H"], (1080, 1920), focal=1380)
        joints = body({"lh": np.array([1.6, 2.2, 0.1]), "rh": np.array([2.0, 2.0, 0.1]),
                       "lf": np.array([1.5, 0.9, 0.05]), "rf": np.array([2.1, 0.8, 0.05])})
        skeleton = joints @ camera["R"].T + camera["t"]
        skeleton -= skeleton[23]  # any origin: the lift solves the translation
        lifted = lift_skeleton(camera, project(camera, joints), np.ones(33), skeleton)
        np.testing.assert_allclose(lifted["board"], joints, atol=1e-6)


@unittest.skipIf(cv2 is None, "needs the wireframe extra")
class SyntheticVideoTest(unittest.TestCase):
    def test_pipeline_recovers_board_and_climber(self):
        from wireframe.extract import PoseFile, evaluate, extract
        from wireframe.geometry import find_climb
        from wireframe.synthetic import make
        climb = find_climb("Lack of Faith", 40)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make(climb, root, size=(720, 1280), fps=12)  # Instagram resolution
            result = extract(root / "video.mp4", climb, pose_source=PoseFile(root / "pose.json"), progress=False)
            report = evaluate(result, json.loads((root / "truth.json").read_text()))
        times = [f["t"] for f in result["frames"]]
        np.testing.assert_allclose(np.diff(times), 0.25, atol=1 / 12)
        self.assertGreaterEqual(result["registration"]["registered"], 0.9 * len(result["frames"]))
        self.assertTrue(result["camera"]["static"])
        self.assertAlmostEqual(report["focal_px"]["estimated"] / report["focal_px"]["true"], 1, delta=0.03)
        self.assertLess(report["climb_holds_reprojection_px"]["mean"], 6)
        self.assertLess(report["joint_error_m"]["mean"], 0.12)
        self.assertLess(report["wall_distance_error_m"]["mean"], 0.08)
        self.assertTrue(any(f["contacts"] for f in result["frames"] if f["pose"]))


if __name__ == "__main__":
    unittest.main()
