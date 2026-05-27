"""
=============================================================
  Unified Inference Script — All Configurations

  Handles all 4 detector + pipeline combinations,
  both source-only and DAOD weights.

  Usage:
    python inference.py --model yolov5n_mog   --mode source
    python inference.py --model yolov5n_mog   --mode daod
    python inference.py --model yolov8s_raw   --mode source
    python inference.py --model yolov8s_raw   --mode daod
    python inference.py --model yolov26s_mog  --mode source
    python inference.py --model yolov26s_mog  --mode daod
    python inference.py --model yolov26s_raw  --mode source
    python inference.py --model yolov26s_raw  --mode daod

  Outputs:
    grid_{model}_{mode}.mp4      — 2x2 grid video
    tracking_{model}_{mode}.mp4  — clean tracking video

  Grid layout:
    [ Preprocessed / Ch3 BGR  |  TVG / Ch3 Extracted ]
    [ MOG2 Motion / Ch3 Input |  KC Tracker + LOI     ]
=============================================================
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from loi_counter import LOICounter

# ── Paths ───────────────────────────────────────────────────
from config import BASE_DIR
# Path to the input video file for inference
VIDEO_PATH = os.environ.get("SONAR_VIDEO_PATH", "merged_output.mp4")

WEIGHTS = {
    "yolov5n_mog": {
        "source": os.path.join(BASE_DIR, "runs", "detect",
                               "yolov5n_pre_pro_run_100e", "weights", "best.pt"),
        "daod":   os.path.join(BASE_DIR, "daod", "weights",
                               "yolov5n_cfc_mog2tvg_daod", "weights", "best.pt"),
        "pipeline": "mog2tvg",
        "label":    "YOLOv5n + MOG2/TVG",
    },
    "yolov8s_raw": {
        "source": os.path.join(BASE_DIR, "runs", "detect",
                               "yolov8s_cfc_raw", "weights", "best.pt"),
        "daod":   os.path.join(BASE_DIR, "daod", "weights",
                               "yolov8s_cfc_raw_daod", "weights", "best.pt"),
        "pipeline": "raw",
        "label":    "YOLOv8s + Raw",
    },
    "yolov26s_mog": {
        "source": os.path.join(BASE_DIR, "runs", "detect",
                               "yolov26smog_cfc_mog2tvgtune", "weights", "best.pt"),
        "daod":   os.path.join(BASE_DIR, "daod", "weights",
                               "yolov26smog_cfc_mog2tvg_daod", "weights", "best.pt"),
        "pipeline": "mog2tvg",
        "label":    "YOLOv26s + MOG2/TVG",
    },
    "yolov26s_raw": {
        "source": os.path.join(BASE_DIR, "runs", "detect",
                               "yolov26s_cfc_rawtune", "weights", "best.pt"),
        "daod":   os.path.join(BASE_DIR, "daod", "weights",
                               "yolov26s_cfc_raw_daod", "weights", "best.pt"),
        "pipeline": "raw",
        "label":    "YOLOv26s + Raw",
    },
}

# ── Inference params ────────────────────────────────────────
CONF         = 0.25
IMG_SIZE     = 512
DEVICE       = 0 if torch.cuda.is_available() else "cpu"
MOG2_HISTORY = 200
BLUR_KERNEL  = (5, 5)
CELL_H       = 640

# TVG params
TVG_SLOPE  = 0.0005
TVG_OFFSET = 1e-3
TVG_MAX_DB = 50.0
TVG_ATTN   = 0.04

# Kalman Centroid Tracker params
KCT_MAX_AGE  = 15
KCT_MIN_HITS = 1
KCT_MAX_DIST = 120.0

ID_COLORS = [
    (255, 80,  80),  (80, 255,  80),  (80,  80, 255), (255, 255,  80),
    (255, 80, 255),  (80, 255, 255), (255, 160,  80), (160,  80, 255),
]
def id_color(tid): return ID_COLORS[(tid - 1) % len(ID_COLORS)]


# ── Kalman Centroid Tracker ─────────────────────────────────
class KCTrack:
    count = 0
    def __init__(self, bbox):
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.Q = np.diag([1., 1., 0.5, 0.5]).astype(np.float32)
        self.R = np.diag([4., 4.]).astype(np.float32)
        self.P = np.diag([10., 10., 100., 100.]).astype(np.float32)
        self.x = np.array([cx, cy, 0., 0.], dtype=np.float32)
        KCTrack.count += 1
        self.id   = KCTrack.count
        self.bbox = list(bbox)
        self.hits = 1; self.hit_streak = 1; self.time_since_update = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0

    def update(self, bbox):
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        z = np.array([cx, cy], dtype=np.float32)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x    = self.x + K @ (z - self.H @ self.x)
        self.P    = (np.eye(4) - K @ self.H) @ self.P
        self.bbox = list(bbox)
        self.hits += 1; self.hit_streak += 1; self.time_since_update = 0

    def centroid(self):
        return float(self.x[0]), float(self.x[1])


class KCTracker:
    def __init__(self, max_age, min_hits, max_dist):
        self.max_age  = max_age
        self.min_hits = min_hits
        self.max_dist = max_dist
        self.tracks   = []
        self.frame_count = 0

    def reset(self):
        self.tracks = []; self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1
        for t in self.tracks:
            t.predict()

        ud = set(range(len(dets)))
        ut = set(range(len(self.tracks)))

        if dets and self.tracks:
            dc   = np.array([((d[0]+d[2])/2, (d[1]+d[3])/2) for d in dets], dtype=np.float32)
            tc   = np.array([t.centroid() for t in self.tracks],              dtype=np.float32)
            dist = np.sqrt(((dc[:, None, :] - tc[None, :, :]) ** 2).sum(-1))
            pairs = sorted(
                [(dist[d, t], d, t) for d in range(len(dets)) for t in range(len(self.tracks))],
                key=lambda x: x[0]
            )
            for dd, di, ti in pairs:
                if di not in ud or ti not in ut or dd > self.max_dist:
                    continue
                self.tracks[ti].update(dets[di])
                ud.discard(di); ut.discard(ti)

        for di in ud:
            self.tracks.append(KCTrack(dets[di]))

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks
                if t.time_since_update == 0 and
                (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)]


# ── Preprocessing ───────────────────────────────────────────
def apply_tvg(gray):
    h, w = gray.shape
    x    = np.arange(w, dtype=np.float32)
    R    = np.tile(x, (h, 1)) * TVG_SLOPE + TVG_OFFSET
    tvg_lin = 10 ** (np.clip(20 * np.log10(R) + TVG_ATTN * R, 0, TVG_MAX_DB) / 20.0)
    return np.clip(gray.astype(np.float32) * tvg_lin, 0, 255).astype(np.uint8)


def preprocess_mog2tvg(frame_bgr, back_sub, prev_tvg):
    """Full MOG2+TVG preprocessing — returns (3ch tensor, tvg_gray, fg_mask)."""
    blurred = cv2.GaussianBlur(frame_bgr, BLUR_KERNEL, 0)
    gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    tvg     = apply_tvg(gray)
    fg      = back_sub.apply(tvg)
    bg      = back_sub.getBackgroundImage()
    if bg is None:
        bg = np.zeros_like(tvg)
    if len(bg.shape) == 3:
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    motion = cv2.absdiff(tvg, prev_tvg) if prev_tvg is not None else np.zeros_like(tvg)
    return cv2.merge([fg, bg, motion]), tvg, fg


def preprocess_raw(frame_bgr):
    """Raw Ch3 extraction — returns (ch3_bgr, ch3_gray, ch3_bgr)."""
    ch3     = frame_bgr[:, :, 2]
    ch3_bgr = cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR)
    return ch3_bgr, ch3, ch3_bgr


# ── Draw helpers ────────────────────────────────────────────
def letterbox(img, tw, th):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    sh, sw = img.shape[:2]
    s      = min(tw / sw, th / sh)
    nw, nh = int(sw * s), int(sh * s)
    res    = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out    = np.zeros((th, tw, 3), dtype=np.uint8)
    px, py = (tw - nw) // 2, (th - nh) // 2
    out[py:py+nh, px:px+nw] = res
    return out, s, px, py


def draw_tracks(frame, active, s, px, py):
    for t in active:
        x1, y1, x2, y2 = t.bbox
        c   = id_color(t.id)
        dx1 = int(x1 * s + px); dy1 = int(y1 * s + py)
        dx2 = int(x2 * s + px); dy2 = int(y2 * s + py)
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), c, 2)
        lbl_txt = f"ID:{t.id}"
        (tw2, th2), _ = cv2.getTextSize(lbl_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (dx1, dy1 - th2 - 6), (dx1 + tw2 + 4, dy1), c, -1)
        cv2.putText(frame, lbl_txt, (dx1 + 2, dy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, pred_count, total_unique, frame_num,
             fps=0.0, inf_ms=0.0, loi_counts=None):
    h, w  = frame.shape[:2]
    ov    = frame.copy()
    bar_h = 70
    cv2.rectangle(ov, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"In frame: {pred_count}", (10, h - bar_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Unique IDs: {total_unique}", (w // 3, h - bar_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 80), 2, cv2.LINE_AA)
    loi_total = loi_counts["total"]       if loi_counts else 0
    loi_r     = loi_counts["count_right"] if loi_counts else 0
    loi_l     = loi_counts["count_left"]  if loi_counts else 0
    cv2.putText(frame, f"LOI R:{loi_r} L:{loi_l} T:{loi_total}",
                (10, h - bar_h + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} FPS  {inf_ms:.1f}ms",
                (w // 2, h - bar_h + 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"F:{frame_num}", (w - 80, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
    return frame


def add_label(img, text):
    cv2.putText(img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img


# ── Main ────────────────────────────────────────────────────
def run(model_key, mode):
    cfg         = WEIGHTS[model_key]
    weight_path = cfg[mode]
    pipeline    = cfg["pipeline"]
    det_label   = f"{cfg['label']} | {'DAOD' if mode == 'daod' else 'Source-Only'}"

    grid_out  = f"grid_{model_key}_{mode}.mp4"
    track_out = f"tracking_{model_key}_{mode}.mp4"

    print(f"\n{'='*60}")
    print(f"  Model    : {model_key}  |  Mode: {mode.upper()}")
    print(f"  Pipeline : {pipeline}")
    print(f"  Weights  : {weight_path}")
    print(f"{'='*60}")

    if not Path(weight_path).exists():
        print(f"ERROR: Weights not found: {weight_path}")
        return

    model   = YOLO(weight_path)
    tracker = KCTracker(KCT_MAX_AGE, KCT_MIN_HITS, KCT_MAX_DIST)
    KCTrack.count = 0  # reset global ID counter

    # MOG2 only for mog2tvg pipeline
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, detectShadows=True
    ) if pipeline == "mog2tvg" else None
    prev_tvg = None

    global_ids  = set()
    frame_num   = 0
    frame_times = []
    fps_avg     = 0.0
    inf_ms_avg  = 0.0

    # Read sample frame for dimensions
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
    ret, sample = cap.read(); cap.release()
    if not ret:
        print("ERROR: Cannot read video"); return
    sh, sw = sample.shape[:2]

    loi = LOICounter(frame_width=sw, line_ratio=0.5)
    loi._last_pos = {}

    CELL_W = int(CELL_H * sw / sh); CELL_W += CELL_W % 2
    fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
    grid_writer  = cv2.VideoWriter(grid_out,  fourcc, fps, (CELL_W * 2, CELL_H * 2))
    track_writer = cv2.VideoWriter(track_out, fourcc, fps, (CELL_W, CELL_H))

    cap = cv2.VideoCapture(VIDEO_PATH)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # ── Preprocess ──────────────────────────────────
        if pipeline == "mog2tvg":
            model_input, tvg_frame, fg_mask = preprocess_mog2tvg(
                frame, back_sub, prev_tvg)
            prev_tvg = tvg_frame.copy()
            v1_img   = model_input   # preprocessed 3ch
            v2_img   = tvg_frame     # TVG gray
            v3_img   = fg_mask       # MOG2 motion mask
            v1_lbl   = "Preprocessed (MOG2+TVG)"
            v2_lbl   = "TVG Frame"
            v3_lbl   = "MOG2 Motion"
        else:
            model_input, ch3_gray, ch3_bgr = preprocess_raw(frame)
            v1_img = frame           # original frame
            v2_img = ch3_gray        # Ch3 extracted
            v3_img = ch3_bgr         # Ch3 BGR (model input)
            v1_lbl = "Original Frame"
            v2_lbl = "Ch3 Extracted"
            v3_lbl = "Ch3 BGR (model input)"

        # ── Detect ──────────────────────────────────────
        t_start = time.perf_counter()
        results = model.predict(
            model_input, conf=CONF, imgsz=IMG_SIZE,
            verbose=False, device=DEVICE
        )[0]
        pred_count = len(results.boxes) if results.boxes is not None else 0
        inf_ms     = (time.perf_counter() - t_start) * 1000

        frame_times.append(inf_ms)
        if len(frame_times) > 30:
            frame_times.pop(0)
        inf_ms_avg = sum(frame_times) / len(frame_times)
        fps_avg    = 1000.0 / inf_ms_avg if inf_ms_avg > 0 else 0

        # ── Track ────────────────────────────────────────
        dets = []
        if pred_count > 0:
            for box in results.boxes.xyxy.cpu().numpy():
                dets.append(list(box))
        active = tracker.update(dets)
        for t in active:
            global_ids.add(t.id)

        # ── LOI ──────────────────────────────────────────
        loi.update_kc(active)

        # ── Grid video ───────────────────────────────────
        v1, _,  _,  _  = letterbox(v1_img, CELL_W, CELL_H)
        v2, _,  _,  _  = letterbox(v2_img, CELL_W, CELL_H)
        v3, _,  _,  _  = letterbox(v3_img, CELL_W, CELL_H)
        v4, s, px, py  = letterbox(frame,  CELL_W, CELL_H)

        add_label(v1, v1_lbl)
        add_label(v2, v2_lbl)
        add_label(v3, v3_lbl)
        add_label(v4, f"KC Tracker | {det_label}")

        v4 = draw_tracks(v4, active, s, px, py)
        loi.draw_loi(v4, scale=s, pad_x=px, pad_y=py)
        v4 = draw_hud(v4, pred_count, len(global_ids), frame_num,
                      fps=fps_avg, inf_ms=inf_ms_avg, loi_counts=loi.get_counts())

        grid = np.vstack([np.hstack([v1, v2]), np.hstack([v3, v4])])
        grid_writer.write(grid)

        # ── Tracking video ───────────────────────────────
        clean, s, px, py = letterbox(frame, CELL_W, CELL_H)
        clean = draw_tracks(clean, active, s, px, py)
        loi.draw_loi(clean, scale=s, pad_x=px, pad_y=py)
        clean = draw_hud(clean, pred_count, len(global_ids), frame_num,
                         fps=fps_avg, inf_ms=inf_ms_avg, loi_counts=loi.get_counts())
        add_label(clean, det_label)
        track_writer.write(clean)

        if frame_num % 100 == 0:
            print(f"  Frame {frame_num:5d} | dets:{pred_count:3d} | "
                  f"active:{len(active):3d} | unique:{len(global_ids):4d} | "
                  f"{fps_avg:.1f} FPS")

    loi.finalize()
    cap.release(); grid_writer.release(); track_writer.release()
    counts = loi.get_counts()

    print(f"\n{'='*60}")
    print(f"  [DONE] {model_key} | {mode.upper()}")
    print(f"  Grid video       : {grid_out}")
    print(f"  Tracking video   : {track_out}")
    print(f"  Unique fish IDs  : {len(global_ids)}")
    print(f"  LOI right        : {counts['count_right']}")
    print(f"  LOI left         : {counts['count_left']}")
    print(f"  LOI total        : {counts['total']}")
    print(f"  Avg FPS          : {fps_avg:.2f}")
    print(f"  Avg inf ms       : {inf_ms_avg:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified inference — all 4 detector configurations"
    )
    parser.add_argument(
        "--model",
        choices=["yolov5n_mog", "yolov8s_raw", "yolov26s_mog", "yolov26s_raw"],
        required=True,
        help="Detector + pipeline combination"
    )
    parser.add_argument(
        "--mode",
        choices=["source", "daod"],
        required=True,
        help="source = source-only weights | daod = DAOD fine-tuned weights"
    )
    args = parser.parse_args()
    run(args.model, args.mode)
