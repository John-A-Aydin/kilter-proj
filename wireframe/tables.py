"""Flat tables from a wireframe result, for dashboards and analysis.

Writes JSON arrays of row objects:
    joints.json    one row per sample, joint and source (estimate, and truth when given),
                   with board (x, s, d) and world (X, Y, Z) coordinates in metres
    holds.json     every hole on the board, with the climb's role where the climb uses it
    samples.json   one row per sample: board found, contacts, distances from the wall
    contacts.json  one row per joint touching a hold
    metrics.json   summary numbers: registration, camera, and errors against the truth
    frames.json    with images: each sampled video frame as a small JPEG data URL

Estimated joints and the climb's holds also carry their image position (u, v) in the
video's own pixels, so a viewer can draw the fit over the frame.
"""
import base64
import json
from pathlib import Path

import numpy as np

from .geometry import ROLES, board_to_world, project
from .pose import JOINTS

LIMBS = {"hips": (23, 24), "left_hand": (15,), "right_hand": (16,), "left_foot": (31,), "right_foot": (32,)}


def build(result, truth=None):
    angle = result["climb"]["angle"]
    truth_frames = {int(k): np.array(v) for k, v in (truth or {}).get("joints_board", {}).items()}
    joints, samples, contacts = [], [], []
    for frame in result["frames"]:
        t = frame["t"]
        sources = []
        if frame["pose"]:
            sources.append(("estimate", np.array(frame["board"])))
        if frame["frame"] in truth_frames:
            sources.append(("truth", truth_frames[frame["frame"]]))
        for source, board in sources:
            world = board_to_world(board, angle)
            pixels = frame.get("pixels") if source == "estimate" else None
            for j, name in enumerate(JOINTS):
                row = {"t": t, "joint": name, "source": source,
                       "x": round(float(board[j][0]), 4), "s": round(float(board[j][1]), 4),
                       "d": round(float(board[j][2]), 4), "X": round(float(world[j][0]), 4),
                       "Y": round(float(world[j][1]), 4), "Z": round(float(world[j][2]), 4)}
                if pixels:
                    row["u"], row["v"] = round(float(pixels[j][0]), 1), round(float(pixels[j][1]), 1)
                    row["visibility"] = round(float(frame["visibility"][j]), 3)
                joints.append(row)
        row = {"t": t, "registered": frame["registered"], "lit_holds": frame["inliers"], "pose": frame["pose"],
               "contacts": len(frame.get("contacts", []))}
        if frame["pose"]:
            board = np.array(frame["board"])
            for limb, ids in LIMBS.items():
                row[f"{limb}_wall_m"] = round(float(board[list(ids), 2].mean()), 4)
            row["skeleton_reprojection_px"] = frame["skeleton_reprojection_px"]
        if frame["frame"] in truth_frames:
            board = truth_frames[frame["frame"]]
            for limb, ids in LIMBS.items():
                row[f"{limb}_wall_true_m"] = round(float(board[list(ids), 2].mean()), 4)
        samples.append(row)
        roles = {h["placement"]: h["role_name"] for h in result["climb"]["holds"]}
        for contact in frame.get("contacts", []):
            contacts.append({"t": t, "joint": contact["joint"], "placement": contact["placement"],
                             "role": roles[contact["placement"]], "gap_cm": round(100 * contact["gap_m"], 1)})

    climb_roles = {h["placement"]: ROLES[h["role"]] for h in result["climb"]["holds"]}
    camera = {key: np.array(result["camera"][key]) for key in ("K", "R", "t")}
    holds = []
    for hole in result["board"]["holes"]:
        world = board_to_world([hole["x"], hole["s"], 0], angle)
        u, v = project(camera, [hole["x"], hole["s"], 0])
        holds.append({"placement": hole["placement"], "role": climb_roles.get(hole["placement"], ""),
                      "x": round(hole["x"], 4), "s": round(hole["s"], 4),
                      "X": round(float(world[0]), 4), "Y": round(float(world[1]), 4), "Z": round(float(world[2]), 4),
                      "u": round(float(u), 1), "v": round(float(v), 1)})

    metrics = []
    def add(group, metric, value, unit=""):
        metrics.append({"group": group, "metric": metric, "value": None if value is None else round(float(value), 4),
                        "unit": unit})
    climb = result["climb"]
    add("climb", "angle", angle, "deg")
    add("video", "interval", result["interval"], "s")
    add("video", "samples", len(result["frames"]))
    add("video", "width", result["image_size"][0], "px")
    add("video", "height", result["image_size"][1], "px")
    add("registration", "frames_with_board", result["registration"]["registered"])
    add("registration", "mean_lit_holds", result["registration"]["mean_inliers"])
    add("registration", "climb_holds", len(climb["holds"]))
    add("pose", "frames_with_wireframe", sum(f["pose"] for f in result["frames"]))
    add("camera", "focal_estimate", result["camera"]["focal_px"], "px")
    for axis, value in zip("XYZ", result["camera"]["center_world"]):
        add("camera", f"estimate_{axis}", value, "m")
    if truth:
        add("camera", "focal_truth", truth["camera"]["focal_px"], "px")
        for axis, value in zip("XYZ", truth["camera"]["center_world"]):
            add("camera", f"truth_{axis}", value, "m")
    evaluation = result.get("evaluation")
    if evaluation:
        add("error", "joint_mean", 100 * evaluation["joint_error_m"]["mean"], "cm")
        add("error", "joint_median", 100 * evaluation["joint_error_m"]["median"], "cm")
        add("error", "joint_p90", 100 * evaluation["joint_error_m"]["p90"], "cm")
        add("error", "wall_distance_mean", 100 * evaluation["wall_distance_error_m"]["mean"], "cm")
        add("error", "climb_holds_reprojection", evaluation["climb_holds_reprojection_px"]["mean"], "px")
        add("error", "camera_position", 100 * evaluation["camera_position_error_m"], "cm")
        for joint, value in evaluation["per_joint_mean_error_m"].items():
            add("joint_error", joint, 100 * value, "cm")
    return {"joints": joints, "holds": holds, "samples": samples, "contacts": contacts, "metrics": metrics}


def frame_images(result, width=360, quality=72):
    """Each sampled frame, downscaled to `width` pixels across, as a JPEG data URL."""
    import cv2
    from .extract import sample_frames
    wanted = {f["frame"] for f in result["frames"]}
    rows = []
    for time, index, frame, _ in sample_frames(result["video"], result["interval"]):
        if index not in wanted:
            continue
        height = round(frame.shape[0] * width / frame.shape[1])
        small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        rows.append({"t": round(time, 4), "frame": index, "width": width, "height": height,
                     "image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()})
    return rows


def write(result_path, output_dir, truth_path=None, images=False):
    result = json.loads(Path(result_path).read_text())
    truth = json.loads(Path(truth_path).read_text()) if truth_path else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build(result, truth)
    if images:
        tables["frames"] = frame_images(result)
    for name, rows in tables.items():
        (output_dir / f"{name}.json").write_text(json.dumps(rows, separators=(",", ":")))
    return {name: len(rows) for name, rows in tables.items()}
