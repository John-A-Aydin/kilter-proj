"""Find the board in a frame from the climb's lit LEDs.

A Kilter beta video shows the climb lit in role colours: start green, hand cyan,
finish magenta, foot orange. Each detected blob is a candidate for one of the
climb's holds of that role. Role-aware RANSAC picks four blob-to-hold matches,
fits a homography from the board plane to the image, and keeps the hypothesis
that explains the most blobs. The winner is refined on all of its matches.
"""
import cv2
import numpy as np

from .geometry import FINISH, FOOT, HAND, START

# OpenCV hue (0-180) at the centre of each role's LED colour, and the accepted spread.
ROLE_HUES = {START: 60, HAND: 91, FINISH: 150, FOOT: 20}
HUE_SPREAD = 13
INLIER_METRES = 2.5 * 0.0254  # holes are at least 5.66 in apart


def detect_leds(frame_bgr, min_saturation=110, min_value=150, return_masks=False):
    """Coloured LED blobs: list of dicts with pixel centre, role, and area.

    With return_masks, also returns {role: per-pixel brightness of that role's colour},
    which `register` uses to refine each hold's centre."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bright = (s >= min_saturation) & (v >= min_value)
    height, width = h.shape
    kernel = np.ones((3, 3), np.uint8)
    # One LED is one brightness peak of its colour. Peaks, unlike connected regions, keep
    # neighbouring LEDs apart when their halos touch, and a halo split by a white core or a
    # hold in front still has one peak once smoothed.
    diagonal = np.hypot(height, width)
    sigma = max(1.5, 0.0025 * diagonal)
    size = max(5, int(round(0.011 * diagonal)) | 1)
    neighbourhood = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    blobs, masks = [], {}
    for role, hue in ROLE_HUES.items():
        distance = np.minimum(np.abs(h.astype(int) - hue), 180 - np.abs(h.astype(int) - hue))
        mask = (bright & (distance <= HUE_SPREAD)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # A bright LED blows out to near-white, leaving only a coloured ring: fill each
        # region's convex hull so the ring becomes a disc centred on the hold.
        filled = np.zeros_like(mask)
        contours, _ = cv2.findContours(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area, hull = cv2.contourArea(contour), cv2.convexHull(contour)
            if area < 3:
                continue
            # A ring is hollow (low solidity); two touching discs are not, and stay apart.
            if area / max(cv2.contourArea(hull), 1e-6) < 0.75:
                cv2.fillConvexPoly(filled, hull, 1)
            else:
                cv2.drawContours(filled, [contour], -1, 1, -1)
        filled |= mask
        weight = filled.astype(np.float32) * 255
        if return_masks:
            masks[role] = weight
        smooth = cv2.GaussianBlur(weight, (0, 0), sigma)
        peaks = ((smooth >= cv2.dilate(smooth, neighbourhood)) & (smooth > 30)).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(peaks)
        for i in range(1, count):
            u, w = centroids[i]
            blobs.append({"u": float(u), "v": float(w), "role": role,
                          "strength": float(smooth[int(round(w)), int(round(u))])})
    return (blobs, masks) if return_masks else blobs


def refine(H, masks, climb, rounds=2):
    """Re-centre each hold on the brightness of its role colour within ~2.5 in, then refit H.

    Returns the refined H and {placement: (u, v)} for every hold with enough lit pixels."""
    points = {}
    for _ in range(rounds):
        src, dst, placements = [], [], []
        for hold in climb["holds"]:
            mask = masks.get(hold["role"])
            if mask is None:
                continue
            p, q = cv2.perspectiveTransform(np.array([[[hold["x"], hold["s"]], [hold["x"] + INLIER_METRES,
                                                                              hold["s"]]]]), H)[0]
            radius = max(4.0, float(np.linalg.norm(q - p)))
            x0, y0 = int(max(0, p[0] - radius)), int(max(0, p[1] - radius))
            x1, y1 = int(min(mask.shape[1], p[0] + radius + 1)), int(min(mask.shape[0], p[1] + radius + 1))
            if x1 <= x0 or y1 <= y0:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            weight = mask[y0:y1, x0:x1] * (((xs - p[0]) ** 2 + (ys - p[1]) ** 2) <= radius ** 2)
            if np.count_nonzero(weight) < 8:
                continue
            total = weight.sum()
            src.append([hold["x"], hold["s"]])
            dst.append([(weight * xs).sum() / total, (weight * ys).sum() / total])
            placements.append(hold["placement"])
        if len(src) < 4:
            break
        src, dst = np.array(src), np.array(dst)
        fitted, _ = cv2.findHomography(src, dst, 0)
        if fitted is None:
            break
        # Drop centres pulled off by occlusion: residuals over 2.5x the median.
        residual = np.linalg.norm(cv2.perspectiveTransform(src[None], fitted)[0] - dst, axis=1)
        keep = residual <= max(2.5 * np.median(residual), 1.5)
        if keep.sum() >= 4 and not keep.all():
            fitted, _ = cv2.findHomography(src[keep], dst[keep], 0)
        H = fitted
        points = {pl: tuple(d) for pl, d, k in zip(placements, dst, keep) if k}
    return H, points


def _match(H_inv, blob_px, blob_roles, hold_xy, hold_roles):
    """One-to-one blob->hold matches within INLIER_METRES, nearest first."""
    mapped = cv2.perspectiveTransform(blob_px[None], H_inv)[0]
    distance = np.linalg.norm(mapped[:, None, :] - hold_xy[None, :, :], axis=2)
    distance[blob_roles[:, None] != hold_roles[None, :]] = np.inf
    pairs = []
    used_blobs, used_holds = set(), set()
    for flat in np.argsort(distance, axis=None):
        b, h = divmod(int(flat), distance.shape[1])
        if distance[b, h] > INLIER_METRES:
            break
        if b not in used_blobs and h not in used_holds:
            pairs.append((b, h, float(distance[b, h])))
            used_blobs.add(b)
            used_holds.add(h)
    return pairs


def _plausible(H, image_size):
    """Board x must run left to right and s bottom to top (image y grows down), at a sane scale."""
    width, height = image_size
    corners = np.array([[[0, 0], [3.66, 0], [3.66, 3.96], [0, 3.96]]], np.float64)
    try:
        projected = cv2.perspectiveTransform(corners, H)[0]
    except cv2.error:
        return False
    if not np.all(np.isfinite(projected)):
        return False
    bottom_left, bottom_right, top_right, top_left = projected
    if not (bottom_right[0] > bottom_left[0] and top_left[1] < bottom_left[1] and top_right[1] < bottom_right[1]):
        return False
    if not cv2.isContourConvex(projected.astype(np.float32)):
        return False
    span = np.ptp(projected, axis=0).max()
    return 0.1 * max(width, height) < span < 12 * max(width, height)


def _support(H, hold_xy, hold_roles, support):
    """How much of each hold's own LED colour lies under its projection: (score, holds lit)."""
    projected = cv2.perspectiveTransform(hold_xy[None], H)[0]
    score, lit = 0.0, 0
    for (u, v), role in zip(projected, hold_roles):
        image = support[role]
        if 0 <= v < image.shape[0] and 0 <= u < image.shape[1]:
            value = float(image[int(v), int(u)])
            score += min(1.0, value / 120)
            lit += value > 40
    return score, lit


def register(blobs, climb, image_size, iterations=6000, seed=0, min_inliers=5, masks=None):
    """Homography from board metres (x, s) to pixels, or None.

    Blobs propose matches; with `masks` from detect_leds, each hypothesis is scored by how
    much of the right colour lies under every projected hold, which does not depend on how
    a halo breaks into blobs. Without masks, the score is the number of matched blobs."""
    if len(blobs) < 4:
        return None
    rng = np.random.default_rng(seed)
    blob_px = np.array([[b["u"], b["v"]] for b in blobs], np.float64)
    blob_roles = np.array([b["role"] for b in blobs])
    hold_xy = np.array([[h["x"], h["s"]] for h in climb["holds"]], np.float64)
    hold_roles = np.array([h["role"] for h in climb["holds"]])
    candidates = {role: np.flatnonzero(hold_roles == role) for role in range(4)}
    usable = np.array([len(candidates[r]) > 0 for r in blob_roles])
    if usable.sum() < 4:
        return None
    support = None
    if masks:
        sigma = max(1.5, 0.005 * np.hypot(*image_size))
        support = {role: cv2.GaussianBlur(mask, (0, 0), sigma) for role, mask in masks.items()}

    def score(H):
        if support is not None:
            return _support(H, hold_xy, hold_roles, support)[0]
        return len(_match(np.linalg.inv(H), blob_px, blob_roles, hold_xy, hold_roles))

    # Favour blobs of rare roles (starts, finishes): fewer wrong assignments to try.
    weights = np.array([1 / len(candidates[r]) if len(candidates[r]) else 0 for r in blob_roles], float)
    weights /= weights.sum()
    best = None
    for _ in range(iterations):
        picked = rng.choice(len(blobs), size=4, replace=False, p=weights)
        holds = []
        for b in picked:
            options = [h for h in candidates[blob_roles[b]] if h not in holds]
            if not options:
                break
            holds.append(rng.choice(options))
        if len(holds) < 4:
            continue
        try:
            H = cv2.getPerspectiveTransform(hold_xy[holds].astype(np.float32), blob_px[picked].astype(np.float32))
        except cv2.error:
            continue
        if not _plausible(H, image_size):
            continue
        value = score(H)
        if best is None or value > best[0]:
            best = (value, H)
            if value >= 0.95 * len(hold_xy):
                break
    if best is None:
        return None
    value, H = best
    for _ in range(3):  # refit on the matched blobs while that does not lower the score
        pairs = _match(np.linalg.inv(H), blob_px, blob_roles, hold_xy, hold_roles)
        if len(pairs) < 4:
            break
        fitted, _ = cv2.findHomography(hold_xy[[p[1] for p in pairs]], blob_px[[p[0] for p in pairs]], 0)
        if fitted is None or not _plausible(fitted, image_size) or score(fitted) < value - 1e-6:
            break
        H, value = fitted, score(fitted)
    points = {}
    if masks:
        refined, points = refine(H, masks, climb)
        if _plausible(refined, image_size) and score(refined) >= value - 0.5:
            H = refined
        lit = _support(H, hold_xy, hold_roles, support)[1]
    else:
        lit = len(_match(np.linalg.inv(H), blob_px, blob_roles, hold_xy, hold_roles))
    if lit < min_inliers:
        return None
    pairs = _match(np.linalg.inv(H), blob_px, blob_roles, hold_xy, hold_roles)
    projected = cv2.perspectiveTransform(hold_xy[None], H)[0]
    residual = [float(np.linalg.norm(projected[climb["holds"].index(next(h for h in climb["holds"]
                if h["placement"] == p))] - np.array(uv))) for p, uv in points.items()]
    return {"H": H, "inliers": int(lit), "blobs": len(blobs), "holds": len(hold_xy),
            "rms_px": float(np.sqrt(np.mean(np.square(residual)))) if residual else None,
            "matches": [{"placement": climb["holds"][h]["placement"], "u": float(blob_px[b][0]),
                         "v": float(blob_px[b][1])} for b, h, _ in pairs],
            "points": points}


def register_from_points(points, holes):
    """Manual fallback: {placement: [u, v]} for at least four holds."""
    src = np.array([holes[int(p)] for p in points], np.float64)
    dst = np.array(list(points.values()), np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    return {"H": H, "inliers": len(points), "blobs": 0, "holds": len(points), "rms_px": None, "matches": [],
            "manual": True}
