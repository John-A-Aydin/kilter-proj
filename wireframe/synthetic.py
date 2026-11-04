"""A synthetic beta video with exact ground truth, for testing the pipeline.

The board is Boardsesh's Kilter Original artwork, placed in 3D at the climb's angle
and filmed by a known camera. The climb's holds glow in their LED colours. A simple
climber moves hand by hand through the climb, with feet following. The video shows
it as a stick figure, which also hides some LEDs, as a real climber does.

Alongside the video it writes the true camera and joints, and MediaPipe-like detections:
true pixels plus noise, and a metric skeleton with noise, a scale error and a small
rotation. The detections stand in for the pose model, which cannot see a stick figure.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import FINISH, FOOT, INCH, START, board_to_world, project, world_to_board
from .pose import JOINTS

ART = Path(__file__).resolve().parent.parent / "web/public/vendor/boardsesh"
TEXTURE_SCALE = 7.5  # artwork pixels per inch (1080 px across 144 in)
LED_BGR = {0: (253, 245, 77), 1: (0, 170, 255), 2: (0, 255, 0), 3: (255, 0, 255)}
J = {name: i for i, name in enumerate(JOINTS)}


def look_at(center_world, target_world, angle, focal, size):
    center, target = world_to_board(center_world, angle), world_to_board(target_world, angle)
    up = world_to_board([0, 1, 0], angle)
    forward = (target - center) / np.linalg.norm(target - center)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    t = -R @ center
    K = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]])
    H = K @ np.stack([R[:, 0], R[:, 1], t], axis=1)
    return {"K": K, "R": R, "t": t, "H": H / H[2, 2], "focal_px": focal, "center_board": center}


def board_texture(climb):
    """Board artwork with the climb lit, in artwork pixels."""
    height, width = 1170, 1080
    panel = np.full((height, width, 3), (38, 36, 40), np.float32)
    for name in ("45-1.webp", "46-1.webp"):
        layer = cv2.imread(str(ART / name), cv2.IMREAD_UNCHANGED).astype(np.float32)
        alpha = layer[..., 3:4] / 255
        panel = panel * (1 - alpha) + layer[..., :3] * alpha
    glow = np.zeros_like(panel)
    for hold in climb["holds"]:
        center = (int(hold["x"] / INCH * TEXTURE_SCALE), int(height - hold["s"] / INCH * TEXTURE_SCALE))
        cv2.circle(glow, center, 15, LED_BGR[hold["role"]], -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 6) * 1.6
    lit = np.clip(panel * 0.55 + glow, 0, 255)
    for hold in climb["holds"]:  # a near-white LED core, as phone cameras record it
        center = (int(hold["x"] / INCH * TEXTURE_SCALE), int(height - hold["s"] / INCH * TEXTURE_SCALE))
        cv2.circle(lit, center, 3, (235, 245, 235), -1, cv2.LINE_AA)
    return lit.astype(np.uint8)


def two_bone(root, end, a, b, pole):
    """Middle joint of a two-bone limb from root to end, bending toward `pole`."""
    axis = end - root
    distance = np.linalg.norm(axis)
    direction = axis / distance
    reach = min(distance, a + b - 1e-3)
    cos_root = np.clip((a * a + reach * reach - b * b) / (2 * a * reach), -1, 1)
    side = pole - direction * (pole @ direction)
    side /= np.linalg.norm(side) + 1e-9
    return root + direction * a * cos_root + side * a * np.sqrt(1 - cos_root ** 2)


def plan_moves(climb):
    """Keyframes of which hold each limb is on: list of dicts with lh, rh, lf, rf board points."""
    holds = climb["holds"]
    def point(h, kind):
        if kind == "hand":
            return np.array([h["x"], h["s"] - 0.06, 0.10])
        return np.array([h["x"], h["s"], 0.05])
    starts = sorted([h for h in holds if h["role"] == START], key=lambda h: h["x"])
    finishes = sorted([h for h in holds if h["role"] == FINISH], key=lambda h: h["x"])
    hands = sorted([h for h in holds if h["role"] == 0], key=lambda h: h["s"])
    feet = sorted([h for h in holds if h["role"] == FOOT], key=lambda h: h["s"]) or sorted(holds, key=lambda h: h["s"])[:2]
    state = {"lh": point(starts[0], "hand"), "rh": point(starts[-1], "hand")}
    low = sorted(feet[:2], key=lambda h: h["x"])
    state["lf"], state["rf"] = point(low[0], "foot"), point(low[-1], "foot")
    keyframes = [dict(state)]
    targets = hands + finishes
    for k, hold in enumerate(targets):
        target = point(hold, "hand")
        if hold in finishes and len(finishes) == 2:
            limb = "lh" if hold is finishes[0] else "rh"
        else:
            limb = "lh" if state["lh"][1] < state["rh"][1] else "rh"
            if abs(state["lh"][1] - state["rh"][1]) < 0.05:
                limb = "lh" if target[0] < (state["lh"][0] + state["rh"][0]) / 2 else "rh"
        state[limb] = target
        keyframes.append(dict(state))
        hands_s = (state["lh"][1] + state["rh"][1]) / 2
        wanted = hands_s - 1.25
        options = [h for h in feet if h["s"] < hands_s - 0.6]
        if options:
            best = min(options, key=lambda h: abs(h["s"] - wanted))
            foot = "lf" if state["lf"][1] < state["rf"][1] else "rf"
            new = point(best, "foot")
            if new[1] > state[foot][1] + 0.05 and not any(np.allclose(new, state[f]) for f in ("lf", "rf")):
                state[foot] = new
                keyframes.append(dict(state))
    if len(finishes) == 1:  # match the single finish
        state["lh"] = state["rh"] = point(finishes[0], "hand") + np.array([-0.03, 0, 0])
        state["rh"] = state["rh"] + np.array([0.06, 0, 0])
        keyframes.append(dict(state))
    return keyframes


def body(limbs):
    """All 33 joints in the board frame from the four end effectors."""
    lh, rh, lf, rf = limbs["lh"], limbs["rh"], limbs["lf"], limbs["rf"]
    hands, feet = (lh + rh) / 2, (lf + rf) / 2
    hip = np.array([(lh[0] + rh[0] + lf[0] + rf[0]) / 4, 0.42 * hands[1] + 0.58 * feet[1], 0.32])
    torso = np.array([hands[0] - hip[0], hands[1] - hip[1], 0])
    torso /= np.linalg.norm(torso) + 1e-9
    across = np.array([torso[1], -torso[0], 0])  # climber's right, in the board plane
    shoulder = hip + torso * 0.48 + np.array([0, 0, 0.04])
    j = np.zeros((33, 3))
    j[J["left_shoulder"]], j[J["right_shoulder"]] = shoulder - across * 0.18, shoulder + across * 0.18
    j[J["left_hip"]], j[J["right_hip"]] = hip - across * 0.10, hip + across * 0.10
    out = np.array([0, 0, 1.0])
    for side, sign, hand, foot in (("left", -1, lh, lf), ("right", 1, rh, rf)):
        s, h = j[J[f"{side}_shoulder"]], j[J[f"{side}_hip"]]
        j[J[f"{side}_wrist"]] = hand
        j[J[f"{side}_elbow"]] = two_bone(s, hand, 0.30, 0.27, across * sign * 0.7 + out - torso * 0.6)
        toe = foot
        ankle = toe + np.array([0, 0.05, 0.13])
        j[J[f"{side}_foot_index"]], j[J[f"{side}_ankle"]] = toe, ankle
        j[J[f"{side}_heel"]] = ankle + np.array([0, -0.02, 0.06])
        j[J[f"{side}_knee"]] = two_bone(h, ankle, 0.44, 0.43, across * sign * 0.8 + out + torso * 0.3)
        for name, offset in (("index", [0, 0.08, -0.04]), ("pinky", [sign * 0.03, 0.06, -0.03]),
                             ("thumb", [-sign * 0.03, 0.05, -0.03])):
            j[J[f"{side}_{name}"]] = hand + np.array(offset)
    head = shoulder + torso * 0.24 + np.array([0, 0, -0.02])
    j[J["nose"]] = head + np.array([0, 0, -0.10])
    for name, offset in (("left_eye_inner", [-0.015, 0.03, -0.09]), ("left_eye", [-0.03, 0.03, -0.085]),
                         ("left_eye_outer", [-0.045, 0.03, -0.08]), ("right_eye_inner", [0.015, 0.03, -0.09]),
                         ("right_eye", [0.03, 0.03, -0.085]), ("right_eye_outer", [0.045, 0.03, -0.08]),
                         ("left_ear", [-0.075, 0.01, 0.0]), ("right_ear", [0.075, 0.01, 0.0]),
                         ("mouth_left", [-0.025, -0.04, -0.09]), ("mouth_right", [0.025, -0.04, -0.09])):
        j[J[name]] = head + np.array(offset)
    return j


def timeline(keyframes, fps, move_seconds=1.0, hold_seconds=0.5):
    """End effectors at every video frame: each move lifts the limb off the wall and back."""
    frames = []
    for _ in range(int(hold_seconds * fps)):
        frames.append(dict(keyframes[0]))
    for previous, current in zip(keyframes, keyframes[1:]):
        for k in range(int(move_seconds * fps)):
            u = min(1.0, k / (0.75 * move_seconds * fps))
            ease = u * u * (3 - 2 * u)
            state = {}
            for limb in ("lh", "rh", "lf", "rf"):
                a, b = previous[limb], current[limb]
                p = a + (b - a) * ease
                if not np.allclose(a, b):
                    p = p + np.array([0, 0, 0.18 * np.sin(np.pi * ease)])
                state[limb] = p
            frames.append(state)
    for _ in range(int(hold_seconds * fps)):
        frames.append(dict(keyframes[-1]))
    return frames


def rotation(rng, degrees):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(degrees)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def make(climb, output_dir, size=(1080, 1920), fps=30, seed=0, camera_world=None, focal=None):
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    angle = climb["angle"]
    focal = focal or 0.72 * max(size)
    target = board_to_world([1.83, 1.85, 0.3], angle)
    camera_world = camera_world if camera_world is not None else np.array([1.65, 1.1, 4.6 + 0.02 * angle])
    camera = look_at(np.asarray(camera_world, float), target, angle, focal, size)
    texture = board_texture(climb)
    A = np.array([[INCH / TEXTURE_SCALE, 0, 0], [0, -INCH / TEXTURE_SCALE, 1170 * INCH / TEXTURE_SCALE], [0, 0, 1]])
    H_texture = camera["H"] @ A
    width, height = size
    board = cv2.warpPerspective(texture, H_texture, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones(texture.shape[:2], np.float32), H_texture, size)[..., None]
    gradient = np.linspace(70, 25, height, dtype=np.float32)[:, None, None]
    room = np.broadcast_to(gradient, (height, width, 3)) * np.array([1.0, 0.97, 0.92], np.float32)
    background = (room * (1 - mask) + board * mask).astype(np.uint8)

    frames = timeline(plan_moves(climb), fps)
    writer = cv2.VideoWriter(str(output_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    truth, detections = {}, {}
    limbs = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24), (11, 23), (12, 24), (23, 25),
             (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]
    skew = rotation(rng, 4.0)
    for index, limbs_state in enumerate(frames):
        joints = body(limbs_state)
        pixels = project(camera, joints)
        image = background.copy()
        for a, b in limbs:
            cv2.line(image, tuple(map(int, pixels[a])), tuple(map(int, pixels[b])), (60, 52, 48), 22, cv2.LINE_AA)
        cv2.circle(image, tuple(map(int, pixels[J["nose"]])), 30, (60, 52, 48), -1, cv2.LINE_AA)
        noise = rng.normal(0, 3, image.shape)
        writer.write(np.clip(image + noise, 0, 255).astype(np.uint8))
        truth[index] = np.round(joints, 5).tolist()
        camera_joints = joints @ camera["R"].T + camera["t"]
        centred = camera_joints - (camera_joints[23] + camera_joints[24]) / 2
        world = 1.06 * centred @ skew.T + rng.normal(0, 0.025, centred.shape)
        visibility = np.full(33, 0.95)
        visibility[:11] = 0.6
        detections[index] = {"pixels": np.round(pixels + rng.normal(0, 2.0, pixels.shape), 2).tolist(),
                             "visibility": visibility.tolist(), "world": np.round(world, 4).tolist()}
    writer.release()
    camera_json = {"H": camera["H"].tolist(), "focal_px": focal, "center_board": camera["center_board"].tolist(),
                   "center_world": list(map(float, camera_world)), "K": camera["K"].tolist(),
                   "R": camera["R"].tolist(), "t": camera["t"].tolist()}
    (output_dir / "truth.json").write_text(json.dumps({"camera": camera_json, "joints_board": truth}))
    (output_dir / "pose.json").write_text(json.dumps({"source": "synthetic", "frames": detections}))
    return {"frames": len(frames), "seconds": len(frames) / fps, "video": str(output_dir / "video.mp4")}
