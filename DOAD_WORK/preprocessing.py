"""
=============================================================
  Sonar Preprocessing Pipelines
  
  Implements TWO pipelines for direct comparison:

  1. Baseline++ (Kay et al., 2022) — OFFLINE
     - No TVG correction
     - Background: clip mean (needs full clip upfront)
     - Frame diff: absdiff(frame, prev_frame) on raw frames
     - 3-channel: [original, bg_subtracted, frame_diff]
     - Cannot run in real-time — requires future frames

  2. MOG2+TVG (Ours, paper 1369) — ONLINE / REAL-TIME
     - TVG correction first (range-dependent gain)
     - Background: MOG2 adaptive model (online, causal)
     - Frame diff: absdiff on TVG-corrected frames
     - 3-channel: [fg_mask, bg_model, motion_mask]
     - Real-time capable — processes frame-by-frame
     - AUV deployable on ZU104 ARM

  Key thesis argument:
    Baseline++ requires the entire clip to compute mean background.
    Our MOG2+TVG pipeline processes each frame causally —
    suitable for real-time AUV deployment where future frames
    are unavailable.

  Additionally, TVG compensates for range-dependent sonar
  signal attenuation — not present in Baseline++.
=============================================================
"""

import cv2
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────
#  TVG Helper (shared, unique to our pipeline)
# ─────────────────────────────────────────────────────────
def apply_tvg(frame_gray, slope=0.0005, offset=1e-3, max_db=50.0, attn=0.04):
    h, w      = frame_gray.shape
    x         = np.arange(w, dtype=np.float32)
    range_map = np.tile(x, (h, 1))
    R         = range_map * slope + offset
    tvg_db    = np.clip(20.0 * np.log10(R) + attn * R, 0.0, max_db)
    tvg_lin   = 10.0 ** (tvg_db / 20.0)
    return np.clip(frame_gray.astype(np.float32) * tvg_lin, 0.0, 255.0).astype(np.uint8)


# ─────────────────────────────────────────────────────────
#  Pipeline 1: Baseline++ — OFFLINE
# ─────────────────────────────────────────────────────────
class BaselinePlusPlusPreprocessor:
    """
    Replicates Baseline++ from Kay et al. (2022).

    3-channel output:
      Ch1: Original grayscale frame
      Ch2: Frame - clip_mean  (background subtracted)
      Ch3: absdiff(frame, prev_frame)  (motion)

    OFFLINE — requires full clip to compute clip_mean.
    """

    def compute_clip_mean(self, img_paths):
        """Compute mean background from all frames. Offline step."""
        accumulator = None
        count       = 0
        for img_path in img_paths:
            frame = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            if accumulator is None:
                accumulator = np.zeros_like(frame, dtype=np.float64)
            accumulator += frame.astype(np.float64)
            count += 1
        if count == 0:
            return None
        return (accumulator / count).astype(np.float32)

    def process_frame(self, frame_gray, clip_mean, prev_frame_gray=None):
        """Process single frame given precomputed clip mean."""
        # Ch1: original
        ch1 = frame_gray
        # Ch2: background subtracted
        ch2 = np.clip(frame_gray.astype(np.float32) - clip_mean, 0, 255).astype(np.uint8)
        # Ch3: frame difference
        ch3 = cv2.absdiff(frame_gray, prev_frame_gray) \
              if prev_frame_gray is not None \
              else np.zeros_like(frame_gray)
        return cv2.merge([ch1, ch2, ch3])

    def process_clip(self, img_paths):
        """Process full clip — reads all frames twice (mean then process)."""
        clip_mean = self.compute_clip_mean(img_paths)
        if clip_mean is None:
            return []
        results         = []
        prev_frame_gray = None
        for img_path in img_paths:
            frame_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if frame_gray is None:
                continue
            output = self.process_frame(frame_gray, clip_mean, prev_frame_gray)
            results.append((img_path, output))
            prev_frame_gray = frame_gray
        return results

    def process_and_save_clip(self, img_paths, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = self.process_clip(img_paths)
        for img_path, output in results:
            cv2.imwrite(str(output_dir / Path(img_path).name), output)
        return len(results)


# ─────────────────────────────────────────────────────────
#  Pipeline 2: MOG2+TVG (Ours) — ONLINE / REAL-TIME
# ─────────────────────────────────────────────────────────
class MOG2TVGPreprocessor:
    """
    Our preprocessing pipeline from paper 1369.
    Exactly matches training preprocessing code.

    3-channel output:
      Ch1: MOG2 foreground mask    (on TVG-corrected frame)
      Ch2: MOG2 background model   (on TVG-corrected frame)
      Ch3: absdiff(tvg_t, tvg_t-1) (motion on TVG frames)

    ONLINE — causal, frame-by-frame, real-time capable.
    AUV deployable on ZU104 ARM.

    Advantages over Baseline++:
      1. TVG corrects range-dependent attenuation
      2. MOG2 adaptive background handles changing conditions
      3. Causal — no future frame dependency
      4. Real-time on ZU104 ARM
    """

    def __init__(self, mog2_history=30, detect_shadows=True,
                 blur_kernel=(5, 5), tvg_slope=0.0005,
                 tvg_offset=1e-3, tvg_max_db=50.0, tvg_attn=0.04):
        self.blur_kernel    = blur_kernel
        self.tvg_slope      = tvg_slope
        self.tvg_offset     = tvg_offset
        self.tvg_max_db     = tvg_max_db
        self.tvg_attn       = tvg_attn
        self.mog2_history   = mog2_history
        self.detect_shadows = detect_shadows
        self._back_sub      = None
        self._prev_tvg      = None
        self._frame_count   = 0
        self.reset()

    def reset(self):
        """Reset MOG2 state between clips. Does NOT reset between frames."""
        self._back_sub = cv2.createBackgroundSubtractorMOG2(
            history       = self.mog2_history,
            detectShadows = self.detect_shadows,
        )
        self._prev_tvg    = None
        self._frame_count = 0

    def extract_raw_from_baseline_pp(self, frame_bgr):
        """
        Extract original sonar frame from Baseline++ preprocessed image.

        Baseline++ channel order (confirmed empirically):
          Ch1 (B): Nearly black — clipped background subtracted
          Ch2 (G): Mean-subtracted frame offset to positive range
          Ch3 (R): Original grayscale sonar frame  ← we want this

        Returns grayscale original sonar frame as np.uint8.
        """
        return frame_bgr[:, :, 2]   # Ch3 = R channel = original

    def process_from_baseline_pp(self, baseline_pp_bgr):
        """
        Process a Baseline++ preprocessed image through our MOG2+TVG pipeline.

        Extracts original sonar frame from Ch3, then applies
        TVG + MOG2 exactly as in paper 1369 training.

        Use this for Channel train/test images which are
        already Baseline++ preprocessed.

        Parameters
        ----------
        baseline_pp_bgr : np.uint8 BGR image loaded from Baseline++ dataset

        Returns
        -------
        output : np.uint8 BGR 3-channel MOG2+TVG preprocessed image
        """
        # Extract original sonar frame from Ch3
        original_gray = self.extract_raw_from_baseline_pp(baseline_pp_bgr)

        # Wrap as BGR for process_frame (which converts to gray internally)
        gray_as_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
        return self.process_frame(gray_as_bgr)

    def process_path_baseline_pp(self, img_path):
        """Load Baseline++ image from path and process through MOG2+TVG."""
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        return self.process_from_baseline_pp(frame)

    def process_clip_baseline_pp(self, img_paths, reset_between=True):
        """
        Process ordered clip of Baseline++ images through MOG2+TVG.
        Resets MOG2 between clips for clean background model.
        """
        if reset_between:
            self.reset()
        results = []
        for img_path in img_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            results.append((img_path, self.process_from_baseline_pp(frame)))
        return results

    def process_and_save_clip_baseline_pp(self, img_paths, output_dir,
                                           reset_between=True):
        """Process Baseline++ clip and save MOG2+TVG results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = self.process_clip_baseline_pp(img_paths, reset_between)
        for img_path, output in results:
            cv2.imwrite(str(output_dir / Path(img_path).name), output)
        return len(results)

    def process_frame(self, frame_bgr):
        """
        Process single frame. Maintains MOG2 state across calls.
        Exactly matches paper 1369 training code.
        """
        self._frame_count += 1

        # Blur + grayscale
        blurred   = cv2.GaussianBlur(frame_bgr, self.blur_kernel, 0)
        gray      = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # TVG — unique to our pipeline
        tvg_frame = apply_tvg(gray, self.tvg_slope, self.tvg_offset,
                              self.tvg_max_db, self.tvg_attn)

        # Ch1: MOG2 foreground mask
        fg_mask  = self._back_sub.apply(tvg_frame)

        # Ch2: MOG2 background model
        bg_model = self._back_sub.getBackgroundImage()
        if bg_model is None:
            bg_model = np.zeros_like(tvg_frame)
        if len(bg_model.shape) == 3:
            bg_model = cv2.cvtColor(bg_model, cv2.COLOR_BGR2GRAY)

        # Ch3: Frame difference on TVG frames
        motion_mask = cv2.absdiff(tvg_frame, self._prev_tvg) \
                      if self._prev_tvg is not None \
                      else np.zeros_like(tvg_frame)

        self._prev_tvg = tvg_frame.copy()

        return cv2.merge([fg_mask, bg_model, motion_mask])

    def process_path(self, img_path):
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        return self.process_frame(frame)

    def process_clip(self, img_paths, reset_between=True):
        if reset_between:
            self.reset()
        results = []
        for img_path in img_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            results.append((img_path, self.process_frame(frame)))
        return results

    def process_and_save_clip(self, img_paths, output_dir, reset_between=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = self.process_clip(img_paths, reset_between)
        for img_path, output in results:
            cv2.imwrite(str(output_dir / Path(img_path).name), output)
        return len(results)


# ─────────────────────────────────────────────────────────
#  Dataset-level batch processor
# ─────────────────────────────────────────────────────────
class DatasetPreprocessor:
    """
    Applies either pipeline to a full dataset.
    Groups images into sequences and processes each as a clip.

    Usage:
        dp = DatasetPreprocessor(pipeline="mog2tvg")
        dp.process_dataset(
            images_dir = "cfc_channel_raw/train",
            output_dir = "daod/target_preprocessed/train",
        )
    """

    def __init__(self, pipeline="mog2tvg", input_format="raw", **kwargs):
        """
        pipeline     : "mog2tvg" or "baseline++"
        input_format : "raw"        -> input images are raw sonar
                       "baseline++" -> input images are Baseline++ preprocessed
                                       (extract Ch3 first, then apply pipeline)
        """
        self.pipeline_name = pipeline
        self.input_format  = input_format
        if pipeline == "mog2tvg":
            self.preprocessor = MOG2TVGPreprocessor(**kwargs)
        elif pipeline == "baseline++":
            self.preprocessor = BaselinePlusPlusPreprocessor()
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

    def process_dataset(self, images_dir, output_dir,
                        extensions=(".jpg", ".png"), verbose=True):
        from utils import build_sequences
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sequences = build_sequences(images_dir, extensions)
        if verbose:
            print(f"[{self.pipeline_name.upper()}] {len(sequences)} sequences")
            print(f"  Input  : {images_dir}")
            print(f"  Output : {output_dir}")
        total = 0
        for seq_idx, (seq_key, img_paths) in enumerate(sequences.items()):
            if self.input_format == "baseline++" and self.pipeline_name == "mog2tvg":
                # Extract Ch3 from Baseline++ then apply MOG2+TVG
                n = self.preprocessor.process_and_save_clip_baseline_pp(
                    img_paths, output_dir
                )
            else:
                n = self.preprocessor.process_and_save_clip(img_paths, output_dir)
            total += n
            if verbose and (seq_idx + 1) % 20 == 0:
                print(f"  Progress: {seq_idx+1}/{len(sequences)} | Processed: {total}")
        if verbose:
            print(f"[{self.pipeline_name.upper()}] Done. Total: {total}")
        return {"pipeline": self.pipeline_name, "total_processed": total}


# ─────────────────────────────────────────────────────────
#  Diagnostic: side-by-side comparison
# ─────────────────────────────────────────────────────────
def compare_pipelines(clip_img_paths, out_path=None, n_frames=3, frame_h=300):
    """
    Generate side-by-side visual comparison of both pipelines.
    Pass a list of image paths from one clip.
    """
    bp_proc  = BaselinePlusPlusPreprocessor()
    mog_proc = MOG2TVGPreprocessor()
    bp_res   = bp_proc.process_clip(clip_img_paths)
    mog_res  = mog_proc.process_clip(clip_img_paths)

    def resize_h(img, h):
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*s), h))

    def labeled_channels(frame_3ch, label, frame_h):
        panels = []
        for i, ch_name in enumerate(["Ch1", "Ch2", "Ch3"]):
            ch = cv2.cvtColor(frame_3ch[:,:,i], cv2.COLOR_GRAY2BGR)
            ch = resize_h(ch, frame_h)
            cv2.putText(ch, f"{label} {ch_name}", (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
            panels.append(ch)
        return np.hstack(panels)

    rows = []
    for i in range(min(n_frames, len(clip_img_paths))):
        orig = cv2.imread(str(clip_img_paths[i]))
        if orig is None:
            continue
        orig_panel = resize_h(orig, frame_h)
        cv2.putText(orig_panel, f"Original f{i}", (4,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
        bp_panel  = labeled_channels(bp_res[i][1],  "BP++", frame_h) \
                    if i < len(bp_res)  else np.zeros((frame_h,1,3), np.uint8)
        mog_panel = labeled_channels(mog_res[i][1], "Ours", frame_h) \
                    if i < len(mog_res) else np.zeros((frame_h,1,3), np.uint8)
        rows.append(np.hstack([orig_panel, bp_panel, mog_panel]))

    if not rows:
        return None

    max_w  = max(r.shape[1] for r in rows)
    padded = [np.hstack([r, np.zeros((r.shape[0], max_w-r.shape[1], 3), np.uint8)])
              for r in rows]
    grid   = np.vstack(padded)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), grid)
        print(f"[VIZ] Comparison saved: {out_path}")
    return grid


# ─────────────────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("=== Preprocessing Pipeline Test ===\n")

    synthetic_bgr  = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    synthetic_gray = cv2.cvtColor(synthetic_bgr, cv2.COLOR_BGR2GRAY)
    clip_mean      = synthetic_gray.astype(np.float32)

    print("Testing Baseline++...")
    bp  = BaselinePlusPlusPreprocessor()
    out = bp.process_frame(synthetic_gray, clip_mean, prev_frame_gray=synthetic_gray)
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8
    print(f"  Output: {out.shape} {out.dtype} ✓")

    print("\nTesting MOG2+TVG (ours)...")
    mog = MOG2TVGPreprocessor()
    out = mog.process_frame(synthetic_bgr)
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8
    print(f"  Output: {out.shape} {out.dtype} ✓")

    print("\nTesting 5 sequential frames...")
    mog.reset()
    for i in range(5):
        mog.process_frame(np.random.randint(50,200,(512,512,3),dtype=np.uint8))
    print(f"  Frame count: {mog._frame_count} ✓")

    if len(sys.argv) > 1:
        clip_dir  = Path(sys.argv[1])
        img_paths = sorted(clip_dir.glob("*.jpg"))[:10]
        if img_paths:
            compare_pipelines(img_paths, "pipeline_comparison.jpg", n_frames=3)

    print("\n[OK] All tests passed")
    print("\nPipeline comparison:")
    print(f"  {'Feature':<30} {'Baseline++ (theirs)':<25} {'MOG2+TVG (ours)'}")
    print(f"  {'-'*75}")
    rows = [
        ("TVG gain correction",    "No",                   "Yes"),
        ("Background method",      "Clip mean (offline)",  "MOG2 adaptive (online)"),
        ("Frame diff input",       "Raw frames",           "TVG-corrected frames"),
        ("Needs full clip?",       "Yes",                  "No"),
        ("Real-time capable?",     "No",                   "Yes"),
        ("AUV deployable?",        "No",                   "Yes"),
    ]
    for feat, bp, ours in rows:
        print(f"  {feat:<30} {bp:<25} {ours}")
