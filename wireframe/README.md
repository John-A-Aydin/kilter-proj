# Climber wireframes from beta videos

A prototype that turns a Kilter beta video into a 3D wireframe of the climber every
0.25 s, in the board's own coordinates: across the board, up the board, and out from
its surface, in metres. Hold positions from the dataset do most of the work.

```bash
uv sync --extra wireframe

# A real video: name the climb and angle as in the dataset.
uv run --extra wireframe python -m wireframe extract beta.mp4 --climb "Lack of Faith" --angle 40 \
  --output runs/wireframe/beta/wireframe.json --overlay-dir runs/wireframe/beta/overlay

# Tables for a dashboard or analysis.
uv run --extra wireframe python -m wireframe tables runs/wireframe/beta/wireframe.json \
  --output-dir runs/wireframe/beta/tables
```

The first run downloads MediaPipe's heavy pose model (about 30 MB) to
`data/cache/models/`. `--overlay-dir` writes each sample with the fitted holds and
skeleton drawn on it: the quickest way to see whether a video worked.

## How it works

1. **Sample** a frame every `--interval` seconds (0.25 by default).
2. **Find the board** (`register.py`). Kilter beta videos show the climb lit: start
   green, hand cyan, finish magenta, foot orange. Each colour's brightness peaks are
   candidate LEDs. RANSAC picks four LED-to-hold matches of the same role, using the
   climb's holds from the dataset, and fits a homography from the board plane to the
   image. A hypothesis scores by how much of the right colour lies under each of the
   climb's holds, which copes with LEDs blown out to white and halos split by a hold.
   Each hold is then re-centred on its colour and the fit refined. When the camera is
   static, every frame's hold positions are pooled into one fit, so holds the climber
   covers in some frames still count.
3. **Recover the camera** (`geometry.py`). The homography gives the focal length,
   assuming square pixels and a centred principal point, and the camera's pose
   relative to the board. A near-frontal view falls back to a typical phone lens, or
   pass `--focal-px`.
4. **Find the body** (`pose.py`). MediaPipe Pose Landmarker gives 33 joints in the
   image and a rough metric 3D skeleton.
5. **Place it** (`geometry.lift_skeleton`). The skeleton is translated along the
   camera's rays so it reprojects onto the detected joints. A wrist or toe whose ray
   meets the board within 12 cm of one of the climb's holds is pinned 8 cm (wrist) or
   6 cm (toe) off the board; these contacts fix the skeleton's depth. Each joint is
   then placed on its own pixel ray at its fitted depth.

The board frame and the world frame for a board overhanging by the climb's angle are
defined in `geometry.py`. The kickboard is treated as part of the main plane, as in the
dataset.

## Synthetic test

No openly licensed Kilter beta video was available, so `python -m wireframe synth`
renders one with exact ground truth: Boardsesh's board artwork placed in 3D at the
climb's angle, the climb lit in LED colours, a known camera, and a simple climber moving
hand by hand. It writes the video, the true camera and joints, and pose detections made
from the truth with noise: 2 px in the image, 2.5 cm per joint, a 6% size error and a 4°
tilt in the metric skeleton. A stick figure can't be detected by MediaPipe, so these
detections replace the pose model in the test.

```bash
uv run --extra wireframe python -m wireframe synth --climb "Lack of Faith" --angle 40 \
  --output-dir runs/wireframe/synthetic
uv run --extra wireframe python -m wireframe extract runs/wireframe/synthetic/video.mp4 \
  --climb "Lack of Faith" --angle 40 --pose-file runs/wireframe/synthetic/pose.json \
  --ground-truth runs/wireframe/synthetic/truth.json --output runs/wireframe/synthetic/wireframe.json
```

Results on that video (Lack of Faith, 40°, 9 s, 36 samples, mean over body joints):

| Resolution | Board found | Joint error | Distance-from-wall error | Focal length |
| --- | ---: | ---: | ---: | ---: |
| 1080×1920 | 36 / 36 | 5.2 cm | 3.0 cm | 1383 px (true 1382) |
| 720×1280 | 36 / 36 | 7.5 cm | 3.8 cm | 917 px (true 922) |
| 540×960 | 29 / 36 | 12.8 cm | 8.1 cm | 644 px (true 691) |

Pinning hands and feet to their holds is what makes depth work: without it, joint error
at 1080×1920 is 21 cm. With perfect board registration it is 4.2 cm, the floor set by
the simulated detection noise. `tests/test_wireframe.py` runs the 720×1280 case.

## Limits

- Tested only on synthetic video. MediaPipe was checked on a real photo of a person,
  not on a climber on a board.
- The climb must be lit and the video must show at least five of its LEDs in some
  frames; otherwise pass `--points` with the pixel position of four or more holds.
- The climber faces the wall, so MediaPipe sees their back; left and right can swap
  on real footage.
- One climber, one board plane, a camera without lens distortion.
