"""
=============================================================
  Inference Benchmark — CPU Speed Test — All 4 Configurations

  Run:
    python inference_benchmark.py --mode yolov5n_mog
    python inference_benchmark.py --mode yolov8s_raw
    python inference_benchmark.py --mode yolov26s_mog
    python inference_benchmark.py --mode yolov26s_raw
    python inference_benchmark.py --mode all

  Output:
    daod/results/inference_benchmark.csv
=============================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time, argparse
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
from preprocessing import MOG2TVGPreprocessor
from utils import get_logger, save_csv

from config import BASE_DIR, LOG_DIR
RESULTS_DIR  = os.path.join(BASE_DIR, "daod", "results")
TEST_IMG_DIR = os.path.join(BASE_DIR, "cfc_channel_test")

CONFIGS = {
    "yolov5n_mog": {
        "weights":  os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov5n_cfc_mog2tvg", "weights", "best.pt"),
        "pipeline": "mog2tvg",
        "note":     "YOLOv5n + MOG2+TVG preprocessing",
    },
    "yolov8s_raw": {
        "weights":  os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov8s_cfc_raw", "weights", "best.pt"),
        "pipeline": "raw",
        "note":     "YOLOv8s + raw Ch3 extraction",
    },
    "yolov26s_mog": {
        "weights":  os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov26s_cfc_mog2tvg", "weights", "best.pt"),
        "pipeline": "mog2tvg",
        "note":     "YOLOv26s + MOG2+TVG preprocessing",
    },
    "yolov26s_raw": {
        "weights":  os.path.join(BASE_DIR, "runs", "detect",
                                 "yolov26s_cfc_raw", "weights", "best.pt"),
        "pipeline": "raw",
        "note":     "YOLOv26s + raw Ch3 extraction",
    },
}

N_WARMUP = 10
N_RUNS   = 100
IMG_SIZE = 512
DEVICE   = "cpu"
CONF     = 0.4


def prepare_input(img_bgr, pipeline, preprocessor):
    ch3 = img_bgr[:, :, 2]
    if pipeline == "mog2tvg":
        return preprocessor.process_frame(cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR))
    return cv2.cvtColor(ch3, cv2.COLOR_GRAY2BGR)


def benchmark_config(mode, cfg, test_images, logger):
    logger.info(f"\n{'='*55}")
    logger.info(f"  Config   : {mode}")
    logger.info(f"  Pipeline : {cfg['pipeline']}")
    logger.info(f"  Device   : CPU")
    logger.info(f"  Runs     : {N_RUNS} (after {N_WARMUP} warmup)")
    logger.info(f"{'='*55}")

    if not Path(cfg["weights"]).exists():
        logger.warning(f"  Weights not found: {cfg['weights']}")
        return None

    model        = YOLO(cfg["weights"])
    preprocessor = MOG2TVGPreprocessor()

    imgs = []
    for img_path in test_images[:N_RUNS + N_WARMUP]:
        frame = cv2.imread(str(img_path))
        if frame is not None:
            imgs.append(frame)
    if not imgs:
        logger.warning("  No test images found")
        return None
    logger.info(f"  Loaded {len(imgs)} test images")

    # Warmup
    logger.info("  Warming up...")
    preprocessor.reset()
    for i in range(min(N_WARMUP, len(imgs))):
        processed = prepare_input(imgs[i], cfg["pipeline"], preprocessor)
        model.predict(source=processed, conf=CONF, imgsz=IMG_SIZE,
                      verbose=False, device=DEVICE)

    # Benchmark preprocessing
    logger.info("  Benchmarking preprocessing...")
    preprocessor.reset()
    pre_times = []; processed_imgs = []
    for i in range(min(N_RUNS, len(imgs))):
        t0 = time.perf_counter()
        processed = prepare_input(imgs[i], cfg["pipeline"], preprocessor)
        pre_times.append((time.perf_counter() - t0) * 1000)
        processed_imgs.append(processed)

    # Benchmark inference
    logger.info("  Benchmarking inference...")
    inf_times = []
    for processed in processed_imgs:
        t0 = time.perf_counter()
        model.predict(source=processed, conf=CONF, imgsz=IMG_SIZE,
                      verbose=False, device=DEVICE)
        inf_times.append((time.perf_counter() - t0) * 1000)

    pre_ms   = float(np.mean(pre_times)); pre_std  = float(np.std(pre_times))
    inf_ms   = float(np.mean(inf_times)); inf_std  = float(np.std(inf_times))
    total_ms = pre_ms + inf_ms
    fps      = 1000.0 / total_ms

    result = {
        "config":         mode,
        "pipeline":       cfg["pipeline"],
        "device":         "cpu",
        "n_runs":         len(inf_times),
        "preprocess_ms":  round(pre_ms,   2),
        "preprocess_std": round(pre_std,  2),
        "inference_ms":   round(inf_ms,   2),
        "inference_std":  round(inf_std,  2),
        "total_ms":       round(total_ms, 2),
        "fps_cpu":        round(fps,      2),
        "note":           cfg["note"],
    }

    logger.info(f"\n  Preprocessing : {pre_ms:.1f} ± {pre_std:.1f} ms")
    logger.info(f"  Inference     : {inf_ms:.1f} ± {inf_std:.1f} ms")
    logger.info(f"  Total         : {total_ms:.1f} ms/frame")
    logger.info(f"  FPS (CPU)     : {fps:.2f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
        choices=["yolov5n_mog","yolov8s_raw","yolov26s_mog","yolov26s_raw","all"],
        required=True)
    args   = parser.parse_args()
    logger = get_logger("inference_benchmark", LOG_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    test_dir    = Path(TEST_IMG_DIR)
    test_images = sorted(list(test_dir.glob("*.jpg")) +
                         list(test_dir.glob("*.png")))
    logger.info(f"  Test images available: {len(test_images)}")

    modes   = list(CONFIGS.keys()) if args.mode == "all" else [args.mode]
    results = []

    for mode in modes:
        r = benchmark_config(mode, CONFIGS[mode], test_images, logger)
        if r:
            results.append(r)

    if results:
        logger.info(f"\n\n{'='*65}")
        logger.info(f"  CPU INFERENCE BENCHMARK RESULTS")
        logger.info(f"{'='*65}")
        logger.info(f"  {'Config':<18} {'Pre(ms)':>8} {'Inf(ms)':>8} "
                    f"{'Total(ms)':>10} {'FPS':>8}")
        logger.info(f"  {'-'*55}")
        for r in results:
            logger.info(f"  {r['config']:<18} "
                        f"{r['preprocess_ms']:>8.1f} "
                        f"{r['inference_ms']:>8.1f} "
                        f"{r['total_ms']:>10.1f} "
                        f"{r['fps_cpu']:>8.2f}")
        logger.info(f"{'='*65}")

        out_path = os.path.join(RESULTS_DIR, "inference_benchmark.csv")
        save_csv(out_path, results)
        logger.info(f"\n  [SAVED] {out_path}")

    logger.info("\n[DONE]")
