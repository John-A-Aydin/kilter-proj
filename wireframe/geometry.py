"""Board frame, climb lookup, camera recovery from a homography, and skeleton lifting.

Board frame (metres), right-handed:
    x  along the board, left to right (0 at the board's left edge)
    s  up the board surface, bottom to top (0 at the bottom of the kickboard)
    d  out of the board surface, into the room (0 on the surface)

The dataset gives every hole on this one plane, in inches (144 x 156 for the 12x12
with kickboard), so the kickboard is treated as part of the main plane.

World frame (metres): X = x, Y up against gravity, Z horizontally out from the wall
base. A board overhanging by `angle` degrees leans its top toward the room.
"""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

INCH = 0.0254
BOARD_WIDTH = 144 * INCH
BOARD_HEIGHT = 156 * INCH
ROLES = ["hand", "foot", "start", "finish"]
HAND, FOOT, START, FINISH = range(4)
ROOT = Path(__file__).resolve().parent.parent
VOCABULARY = ROOT / "data/training/hold_vocabulary.csv"
CLIMBS = ROOT / "data/training/climbs.jsonl.gz"


def load_holes(path=VOCABULARY):
    """Placement ID -> (x, s) in metres on the board plane."""
    with Path(path).open(newline="") as source:
        return {int(r["hold_token"]): (float(r["raw_x"]) * INCH, float(r["raw_y"]) * INCH)
                for r in csv.DictReader(source)}


def find_climb(name, angle, source=CLIMBS, holes=None):
    """The single-frame climb with this name (case-insensitive) at this angle."""
    holes = holes or load_holes()
    wanted = name.strip().casefold()
    matches = []
    with gzip.open(source, "rt", encoding="utf-8") as rows:
        for row in map(json.loads, rows):
            if row["angle"] == angle and row["name"].strip().casefold() == wanted:
                matches.append(row)
    if not matches:
        raise ValueError(f"No climb named {name!r} at {angle}° in {source}")
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} climbs are named {name!r} at {angle}°; pass --uuid")
    return climb_record(matches[0], holes)


def find_climb_by_uuid(uuid, angle, source=CLIMBS, holes=None):
    holes = holes or load_holes()
    with gzip.open(source, "rt", encoding="utf-8") as rows:
        for row in map(json.loads, rows):
            if row["climb_uuid"] == uuid and row["angle"] == angle:
                return climb_record(row, holes)
    raise ValueError(f"No climb {uuid} at {angle}°")


def climb_record(row, holes):
    if len(row["frames"]) != 1:
        raise ValueError("Multi-frame climbs are not supported")
    return {
        "uuid": row["climb_uuid"], "name": row["name"], "setter": row["setter_username"],
        "angle": row["angle"], "grade": row["grade_hueco"],
        "holds": [{"placement": h["hold_token"], "role": h["hold_use_type"],
                   "x": holes[h["hold_token"]][0], "s": holes[h["hold_token"]][1]} for h in row["frames"][0]],
    }


def board_to_world(points, angle):
    """(..., 3) board-frame (x, s, d) points to world (X, Y, Z) for a board overhanging by `angle`°."""
    points = np.asarray(points, dtype=float)
    theta = np.radians(angle)
    x, s, d = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([x, s * np.cos(theta) - d * np.sin(theta), s * np.sin(theta) + d * np.cos(theta)], axis=-1)


def world_to_board(points, angle):
    points = np.asarray(points, dtype=float)
    theta = np.radians(angle)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([x, y * np.cos(theta) + z * np.sin(theta), -y * np.sin(theta) + z * np.cos(theta)], axis=-1)


def estimate_focal(H, image_size):
    """Focal length in pixels from a plane-to-image homography, assuming square pixels and a centred
    principal point. Returns None when the view is too close to fronto-parallel to tell."""
    width, height = image_size
    Hc = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]]) @ H
    h1, h2 = Hc[:, 0], Hc[:, 1]
    # The rotation's first two columns are orthogonal and equally long. With omega = diag(g, g, 1),
    # g = 1/f^2, that gives two equations a_i * g + b_i = 0, solved for g by least squares.
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    if float(a @ a) < 1e-18:
        return None
    g = -float(a @ b) / float(a @ a)
    if g <= 0:
        return None
    focal = 1 / np.sqrt(g)
    long_side = max(width, height)
    return float(focal) if 0.3 * long_side <= focal <= 4 * long_side else None


def camera_from_homography(H, image_size, focal=None):
    """K, R, t with camera = R @ board + t, from H mapping board-plane (x, s, 1) in metres to pixels."""
    width, height = image_size
    source = "given"
    if focal is None:
        focal = estimate_focal(H, image_size)
        source = "homography"
    if focal is None:
        # A typical phone main camera: about 0.7 x the long side of the frame.
        focal, source = 0.7 * max(width, height), "prior"
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    M = np.linalg.inv(K) @ H
    scale = 2 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    M = M * scale
    if M[2, 2] < 0:  # the board must be in front of the camera
        M = -M
    r1, r2, t = M[:, 0], M[:, 1], M[:, 2]
    U, _, Vt = np.linalg.svd(np.stack([r1, r2, np.cross(r1, r2)], axis=1))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    center = -R.T @ t
    return {"K": K, "R": R, "t": t, "focal": float(focal), "focal_source": source, "center_board": center}


def project(camera, board_points):
    """Board-frame points (..., 3) to pixels (..., 2)."""
    P = np.asarray(board_points, dtype=float) @ camera["R"].T + camera["t"]
    uv = P @ camera["K"].T
    return uv[..., :2] / uv[..., 2:3]


def pixel_rays(camera, pixels):
    """Unit rays in the camera frame through each pixel."""
    pixels = np.asarray(pixels, dtype=float)
    homogeneous = np.concatenate([pixels, np.ones(pixels.shape[:-1] + (1,))], axis=-1)
    rays = homogeneous @ np.linalg.inv(camera["K"]).T
    return rays / rays[..., 2:3]  # z = 1, so a depth z gives the point z * ray


def lift_skeleton(camera, pixels, visibility, skeleton, contacts=(), contact_weight=10.0):
    """Place a metric skeleton in the board frame.

    pixels:    (J, 2) detected keypoints.
    skeleton:  (J, 3) metric joint positions in camera-aligned axes, relative to any origin
               (MediaPipe world landmarks: x right, y down, z away from the camera).
    contacts:  (joint, board_point) pairs whose 3D position is known from the board, such as a
               wrist on a hold. They pin the skeleton's distance from the wall.

    Solves the skeleton's translation T in the camera frame by linear least squares:
    each visible joint must reproject onto its pixel, and each contact must sit at its point.
    Each joint is then placed on its own pixel ray at the depth the fitted skeleton gives it,
    so the result reprojects exactly onto the detections.
    """
    K = camera["K"]
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    rows, rhs, weights = [], [], []
    for j, ((u, v), w) in enumerate(zip(pixels, visibility)):
        if w <= 0.3:
            continue
        X, Y, Z = skeleton[j]
        # f (X + Tx) = (u - cx)(Z + Tz), likewise for v; divided by f to keep units in metres.
        rows += [[1, 0, -(u - cx) / f], [0, 1, -(v - cy) / f]]
        rhs += [(u - cx) / f * Z - X, (v - cy) / f * Z - Y]
        weights += [w, w]
    for joint, board_point in contacts:
        target = camera["R"] @ np.asarray(board_point) + camera["t"]
        for axis in range(3):
            row = [0, 0, 0]
            row[axis] = 1
            rows.append(row)
            # Depth equations scale with distance; the 1/z keeps them comparable to the image rows.
            rhs.append(target[axis] - skeleton[joint][axis])
            weights.append(contact_weight / max(target[2], 0.5))
    if len(rows) < 6:
        return None
    A, b, w = np.array(rows, float), np.array(rhs, float), np.sqrt(np.array(weights, float))
    T, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    depth = skeleton[:, 2] + T[2]
    camera_points = pixel_rays(camera, pixels) * depth[:, None]
    board_points = (camera_points - camera["t"]) @ camera["R"]
    reprojected = project(camera, (skeleton + T - camera["t"]) @ camera["R"])
    visible = np.asarray(visibility) > 0.3
    error = float(np.sqrt(np.mean(np.sum((reprojected - pixels) ** 2, axis=1)[visible])))
    return {"board": board_points, "translation": T, "skeleton_reprojection_px": error}


def ray_board_intersection(camera, pixel, offset=0.0):
    """Board-frame point where a pixel's ray meets the plane d = offset."""
    ray_camera = pixel_rays(camera, np.asarray(pixel, float)[None])[0]
    origin = camera["center_board"]
    direction = camera["R"].T @ ray_camera
    if abs(direction[2]) < 1e-9:
        return None
    k = (offset - origin[2]) / direction[2]
    return origin + k * direction if k > 0 else None
