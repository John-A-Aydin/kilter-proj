"""Wireframes from a climbing video, in the board's own coordinates, every `interval` seconds."""

import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    ROLES,
    board_to_world,
    camera_from_homography,
    load_holes,
    project,
    ray_board_intersection,
    lift_skeleton,
)
from .pose import EDGES, FOOT_CONTACTS, HAND_CONTACTS, JOINTS
from .register import detect_leds, register, register_from_points

# How far a touching joint sits off the board surface, and how close (in the plane) to a hold.
WRIST_OFFSET, TOE_OFFSET = 0.08, 0.06
CONTACT_RADIUS = 0.12
CONTACT_MAX_DEPTH = 0.25
ROLE_BGR = {0: (253, 245, 77), 1: (0, 170, 255), 2: (0, 255, 0), 3: (255, 0, 255)}


def sample_frames(path, interval):
    """Yield (time, frame index, BGR frame) every `interval` seconds, reading sequentially."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = {}
    k = 0
    while round(k * interval * fps) < count:
        wanted.setdefault(round(k * interval * fps), k * interval)
        k += 1
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            yield wanted[index], index, frame, fps
        index += 1
    capture.release()


def find_contacts(camera, pixels, visibility, first_pass, climb):
    """Wrists and toes that are on a climb hold: (joint, board point on the joint's ray, placement)."""
    holds = climb["holds"]
    contacts = []
    for joints, offset in ((HAND_CONTACTS, WRIST_OFFSET), (FOOT_CONTACTS, TOE_OFFSET)):
        for joint, roles in joints.items():
            if (
                visibility[joint] <= 0.5
                or first_pass is None
                or first_pass["board"][joint][2] > CONTACT_MAX_DEPTH
            ):
                continue
            point = ray_board_intersection(camera, pixels[joint], offset)
            if point is None:
                continue
            best = None
            for hold in holds:
                if hold["role"] not in roles:
                    continue
                gap = np.hypot(point[0] - hold["x"], point[1] - hold["s"])
                if gap < CONTACT_RADIUS and (best is None or gap < best[0]):
                    best = (gap, hold)
            if best:
                contacts.append((joint, point, best[1]["placement"], best[0]))
    return contacts


def is_static(registrations, image_size, climb):
    """True when most frames put the climb's holds within 1% of the frame diagonal of the best frame.

    Compares the climb's holds, where every fit is constrained, not the board's corners,
    where small errors are extrapolated."""
    good = [r for r in registrations if r]
    if len(good) < 2:
        return True
    best = max(good, key=lambda r: (r["inliers"], -(r["rms_px"] or 0)))
    holds = np.array([[[h["x"], h["s"]] for h in climb["holds"]]])
    reference = cv2.perspectiveTransform(holds, best["H"])[0]
    limit = 0.01 * np.hypot(*image_size)
    moved = [
        np.median(
            np.linalg.norm(
                cv2.perspectiveTransform(holds, r["H"])[0] - reference, axis=1
            )
        )
        >= limit
        for r in good
    ]
    return np.mean(moved) <= 0.2


def draw_overlay(frame, camera, climb, pixels, visibility, contacts):
    image = frame.copy()
    for hold in climb["holds"]:
        u, v = project(camera, [hold["x"], hold["s"], 0])
        cv2.circle(image, (int(u), int(v)), 14, ROLE_BGR[hold["role"]], 2, cv2.LINE_AA)
    if pixels is not None:
        for a, b in EDGES:
            if visibility[a] > 0.3 and visibility[b] > 0.3:
                cv2.line(
                    image,
                    tuple(map(int, pixels[a])),
                    tuple(map(int, pixels[b])),
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
        for joint, *_ in contacts:
            cv2.circle(
                image, tuple(map(int, pixels[joint])), 9, (0, 255, 255), -1, cv2.LINE_AA
            )
    return image


def extract(
    video,
    climb,
    interval=0.25,
    pose_source=None,
    points=None,
    focal=None,
    camera_mode="auto",
    overlay_dir=None,
    progress=True,
):
    """Run the whole pipeline. `pose_source(frame_rgb, time_ms, frame_index)` returns landmarks or None."""
    holes = load_holes()
    samples = list(sample_frames(video, interval))
    if not samples:
        raise ValueError("Video has no frames")
    height, width = samples[0][2].shape[:2]
    fps = samples[0][3]
    image_size = (width, height)

    # 1. The board in each sampled frame.
    if points:
        fixed = register_from_points(points, holes)
        registrations = [fixed] * len(samples)
    else:
        registrations = []
        for i, (time, index, frame, _) in enumerate(samples):
            blobs, masks = detect_leds(frame, return_masks=True)
            registrations.append(
                register(blobs, climb, image_size, seed=index, masks=masks)
            )
            if progress and (i + 1) % 10 == 0:
                print(f"Registered {i + 1}/{len(samples)} frames", flush=True)
    good = [r for r in registrations if r]
    if not good:
        raise ValueError(
            "Could not find the board in any frame: pass --points with four or more holds"
        )
    static = camera_mode == "static" or (
        camera_mode == "auto" and is_static(registrations, image_size, climb)
    )
    if static:
        best = max(good, key=lambda r: (r["inliers"], -(r["rms_px"] or 0)))
        H = best["H"]
        # One camera: each hold's median centre over every frame that saw it, so holds the
        # climber covers in some frames still count.
        seen = {}
        for r in good:
            for placement, uv in r.get("points", {}).items():
                seen.setdefault(placement, []).append(uv)
        if len(seen) >= 4:
            src = np.array([holes[p] for p in seen], np.float64)
            dst = np.array(
                [np.median(np.array(uv), axis=0) for uv in seen.values()], np.float64
            )
            fitted, _ = cv2.findHomography(src, dst, 0)
            if fitted is not None:
                H = fitted
        homographies = [H] * len(samples)
    else:  # moving camera: each frame uses its own board, or the nearest frame that found it
        found = [i for i, r in enumerate(registrations) if r]
        homographies = [
            registrations[min(found, key=lambda j: abs(j - i))]["H"]
            for i in range(len(samples))
        ]

    # 2. Pose, lifted into the board frame.
    frames = []
    overlay_dir = Path(overlay_dir) if overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    for i, (time, index, frame, _) in enumerate(samples):
        camera = camera_from_homography(homographies[i], image_size, focal)
        detection = (
            pose_source(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), time * 1000, index)
            if pose_source
            else None
        )
        record = {
            "t": round(time, 4),
            "frame": index,
            "registered": registrations[i] is not None,
            "inliers": registrations[i]["inliers"] if registrations[i] else 0,
            "blobs": registrations[i]["blobs"] if registrations[i] else 0,
            "pose": False,
        }
        contacts = []
        if detection is not None:
            pixels, visibility, skeleton = (
                detection["pixels"],
                detection["visibility"],
                detection["world"],
            )
            first = lift_skeleton(camera, pixels, visibility, skeleton)
            contacts = find_contacts(camera, pixels, visibility, first, climb)
            lifted = (
                lift_skeleton(
                    camera,
                    pixels,
                    visibility,
                    skeleton,
                    [(joint, point) for joint, point, *_ in contacts],
                )
                if contacts
                else first
            )
            if lifted is not None:
                record.update(
                    {
                        "pose": True,
                        "board": np.round(lifted["board"], 4).tolist(),
                        "world": np.round(
                            board_to_world(lifted["board"], climb["angle"]), 4
                        ).tolist(),
                        "visibility": np.round(visibility, 3).tolist(),
                        "pixels": np.round(pixels, 1).tolist(),
                        "skeleton_reprojection_px": round(
                            lifted["skeleton_reprojection_px"], 2
                        ),
                        "contacts": [
                            {
                                "joint": JOINTS[j],
                                "placement": p,
                                "gap_m": round(float(g), 4),
                            }
                            for j, _, p, g in contacts
                        ],
                    }
                )
        frames.append(record)
        if overlay_dir:
            image = draw_overlay(
                frame,
                camera,
                climb,
                detection["pixels"] if detection else None,
                detection["visibility"] if detection else None,
                contacts,
            )
            cv2.imwrite(
                str(overlay_dir / f"{i:04d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
        if progress and (i + 1) % 10 == 0:
            print(f"Lifted {i + 1}/{len(samples)} frames", flush=True)

    camera = camera_from_homography(homographies[0], image_size, focal)
    return {
        "format": "kilter-wireframe-v1",
        "video": str(video),
        "fps": fps,
        "interval": interval,
        "image_size": [width, height],
        "climb": {
            **climb,
            "holds": [{**h, "role_name": ROLES[h["role"]]} for h in climb["holds"]],
        },
        "board": {
            "width_m": BOARD_WIDTH,
            "height_m": BOARD_HEIGHT,
            "holes": [
                {"placement": p, "x": x, "s": s} for p, (x, s) in sorted(holes.items())
            ],
        },
        "camera": {
            "static": bool(static),
            "focal_px": camera["focal"],
            "focal_source": camera["focal_source"],
            "K": camera["K"].tolist(),
            "R": camera["R"].tolist(),
            "t": camera["t"].tolist(),
            "center_board": camera["center_board"].tolist(),
            "center_world": board_to_world(
                camera["center_board"], climb["angle"]
            ).tolist(),
            "H": homographies[0].tolist(),
        },
        "registration": {
            "frames": len(samples),
            "registered": len(good),
            "mean_inliers": float(np.mean([r["inliers"] for r in good])),
            "rms_px": float(
                np.mean([r["rms_px"] for r in good if r["rms_px"] is not None] or [0])
            ),
        },
        "joints": JOINTS,
        "edges": EDGES,
        "frames": frames,
    }


def evaluate(result, truth):
    """Compare against a synthetic ground truth: per-joint 3D error and camera error."""
    by_frame = {int(k): np.array(v) for k, v in truth["joints_board"].items()}
    body = list(range(11, 33))
    errors, depth_errors, per_joint = [], [], {JOINTS[j]: [] for j in body}
    for frame in result["frames"]:
        if not frame["pose"] or frame["frame"] not in by_frame:
            continue
        estimate, actual = (
            np.array(frame["board"])[body],
            by_frame[frame["frame"]][body],
        )
        error = np.linalg.norm(estimate - actual, axis=1)
        errors.append(error)
        depth_errors.append(np.abs(estimate[:, 2] - actual[:, 2]))
        for j, e in zip(body, error):
            per_joint[JOINTS[j]].append(float(e))
    errors, depth_errors = np.concatenate(errors), np.concatenate(depth_errors)
    true_camera = np.array(truth["camera"]["center_board"])
    holes = np.array([[h["x"], h["s"], 0] for h in result["board"]["holes"]])
    H_true = np.array(truth["camera"]["H"])
    H_est = np.array(result["camera"]["H"])
    proj_true = cv2.perspectiveTransform(holes[None, :, :2], H_true)[0]
    proj_est = cv2.perspectiveTransform(holes[None, :, :2], H_est)[0]
    climb_xy = np.array([[[h["x"], h["s"]] for h in result["climb"]["holds"]]])
    climb_gap = np.linalg.norm(
        cv2.perspectiveTransform(climb_xy, H_true)[0]
        - cv2.perspectiveTransform(climb_xy, H_est)[0],
        axis=1,
    )
    return {
        "joint_error_m": {
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
        },
        "wall_distance_error_m": {
            "mean": float(depth_errors.mean()),
            "median": float(np.median(depth_errors)),
        },
        "per_joint_mean_error_m": {
            k: float(np.mean(v)) for k, v in per_joint.items() if v
        },
        "climb_holds_reprojection_px": {
            "mean": float(climb_gap.mean()),
            "max": float(climb_gap.max()),
        },
        "board_reprojection_px": {
            "mean": float(np.linalg.norm(proj_true - proj_est, axis=1).mean()),
            "max": float(np.linalg.norm(proj_true - proj_est, axis=1).max()),
        },
        "focal_px": {
            "estimated": result["camera"]["focal_px"],
            "true": truth["camera"]["focal_px"],
        },
        "camera_position_error_m": float(
            np.linalg.norm(np.array(result["camera"]["center_board"]) - true_camera)
        ),
    }


class PoseFile:
    """Pre-computed landmarks keyed by frame index (used for the synthetic test)."""

    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        self.frames = {
            int(k): {key: np.array(value) for key, value in v.items()}
            for k, v in data["frames"].items()
        }

    def __call__(self, frame_rgb, time_ms, index):
        return self.frames.get(index)
