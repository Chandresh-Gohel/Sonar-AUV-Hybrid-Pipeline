"""
=============================================================
  Kalman Centroid Tracker
  
  Pure numpy implementation — no filterpy, no scipy.
  Designed for edge deployment on Zynq ZU104 (ARM Cortex-A53).

  Differences from SORT:
  - Centroid distance matching instead of IoU matching
  - Simpler state vector [cx, cy, vx, vy] instead of [x,y,s,r,...]
  - No scipy linear_sum_assignment — uses greedy nearest neighbour
  - Same Kalman predict/update cycle

  Why centroid over IoU for sonar:
  - Sonar fish bboxes are small and irregular
  - IoU between small boxes drops to 0 with tiny position shifts
  - Centroid distance is more stable for sparse sonar detections

  State vector: [cx, cy, vx, vy]
    cx, cy = centroid x, y
    vx, vy = velocity x, y

  Measurement vector: [cx, cy]
=============================================================
"""

import numpy as np


# ── Single Track ───────────────────────────────────────────
class KalmanCentroidTrack:
    """
    Single fish track using Kalman filter on centroid position.
    State: [cx, cy, vx, vy]
    """

    # Global ID counter — never resets so every fish gets unique ID
    count = 0

    def __init__(self, bbox, frame_idx=0):
        """
        bbox: [x1, y1, x2, y2] in pixel coords
        """
        cx, cy = self._bbox_to_centroid(bbox)

        # ── Kalman matrices (4x4 state, 2x4 measurement) ──

        # State transition: constant velocity model
        # [cx]   [1 0 1 0] [cx]
        # [cy] = [0 1 0 1] [cy]
        # [vx]   [0 0 1 0] [vx]
        # [vy]   [0 0 0 1] [vy]
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix: we observe cx, cy only
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise covariance Q — how much we trust the motion model
        # Higher = more responsive to sudden changes
        self.Q = np.diag([1.0, 1.0, 0.5, 0.5]).astype(np.float32)

        # Measurement noise covariance R — how much we trust detections
        # Higher = smoother but slower to respond
        # Sonar detections are noisy so R is moderate
        self.R = np.diag([4.0, 4.0]).astype(np.float32)

        # Initial state covariance P — high uncertainty at start
        self.P = np.diag([10.0, 10.0, 100.0, 100.0]).astype(np.float32)

        # Initial state
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)

        # Track metadata
        self.id               = KalmanCentroidTrack.count + 1
        KalmanCentroidTrack.count += 1

        self.bbox             = bbox          # last known bbox [x1,y1,x2,y2]
        self.hits             = 1             # total detections matched
        self.hit_streak       = 1             # consecutive hits
        self.time_since_update = 0            # frames since last detection
        self.age              = 0             # total frames this track exists
        self.first_frame      = frame_idx     # frame track was born
        self.last_frame       = frame_idx     # frame track was last updated
        self.centroid_history = [(cx, cy)]    # for trajectory visualization

    # ── Kalman predict ─────────────────────────────────────
    def predict(self):
        """
        Predict next state using motion model.
        Called every frame before matching.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        return self.get_centroid()

    # ── Kalman update ──────────────────────────────────────
    def update(self, bbox, frame_idx=0):
        """
        Update state with new detection.
        bbox: [x1, y1, x2, y2]
        """
        cx, cy = self._bbox_to_centroid(bbox)
        z = np.array([cx, cy], dtype=np.float32)

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance (Joseph form for numerical stability)
        I_KH  = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P

        # Update metadata
        self.bbox              = bbox
        self.hits             += 1
        self.hit_streak       += 1
        self.time_since_update = 0
        self.last_frame        = frame_idx
        self.centroid_history.append((cx, cy))

    # ── Helpers ────────────────────────────────────────────
    def get_centroid(self):
        return float(self.x[0]), float(self.x[1])

    def get_velocity(self):
        return float(self.x[2]), float(self.x[3])

    def get_predicted_bbox(self):
        """
        Approximate bbox from predicted centroid using last known size.
        """
        cx, cy = self.get_centroid()
        x1, y1, x2, y2 = self.bbox
        w = x2 - x1
        h = y2 - y1
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

    @staticmethod
    def _bbox_to_centroid(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# ── Multi-Object Tracker ───────────────────────────────────
class KalmanCentroidTracker:
    """
    Multi-object tracker using Kalman centroid tracks.

    Parameters
    ----------
    max_age       : frames a track survives without detection
    min_hits      : minimum detections before track is confirmed
    max_distance  : maximum centroid distance for matching (pixels)
    """

    def __init__(self, max_age=45, min_hits=2, max_distance=50.0):
        self.max_age     = max_age
        self.min_hits    = min_hits
        self.max_distance = max_distance
        self.tracks      = []
        self.frame_count = 0

    def reset(self):
        """
        Reset tracker state between sequences.
        Does NOT reset KalmanCentroidTrack.count — global unique IDs preserved.
        """
        self.tracks      = []
        self.frame_count = 0

    def update(self, detections, frame_idx=0):
        """
        Update tracker with new detections.

        Parameters
        ----------
        detections : list of [x1, y1, x2, y2, conf] or np.array (N,5)
                     Empty array if no detections this frame.
        frame_idx  : current frame number (for metadata)

        Returns
        -------
        active_tracks : list of confirmed tracks
            Each track has .id, .bbox, .hits, .time_since_update etc.
        """
        self.frame_count += 1

        # ── 1. Predict all existing tracks ────────────────
        for track in self.tracks:
            track.predict()

        # ── 2. Match detections to tracks ─────────────────
        if len(detections) == 0:
            dets = []
        else:
            dets = np.array(detections)
            if dets.ndim == 1:
                dets = dets.reshape(1, -1)
            dets = dets[:, :4]   # [x1, y1, x2, y2]

        matched, unmatched_dets, unmatched_trks = \
            self._match(dets, self.tracks)

        # ── 3. Update matched tracks ───────────────────────
        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(dets[det_idx], frame_idx)

        # ── 4. Create new tracks for unmatched detections ──
        for det_idx in unmatched_dets:
            self.tracks.append(
                KalmanCentroidTrack(dets[det_idx].tolist(), frame_idx)
            )

        # ── 5. Remove dead tracks ──────────────────────────
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age]

        # ── 6. Return confirmed active tracks ─────────────
        active = [t for t in self.tracks
                  if t.time_since_update == 0 and
                  (t.hit_streak >= self.min_hits or
                   self.frame_count <= self.min_hits)]

        return active

    # ── Greedy nearest-neighbour matching ─────────────────
    def _match(self, dets, tracks):
        """
        Match detections to tracks using centroid distance.
        Greedy: sort all pairs by distance, assign closest first.
        No scipy needed — pure numpy.
        """
        if len(tracks) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(tracks)))

        # Compute centroid distance matrix (N_dets x N_tracks)
        det_centroids = np.array([
            [(d[0]+d[2])/2., (d[1]+d[3])/2.] for d in dets
        ], dtype=np.float32)

        trk_centroids = np.array([
            list(t.get_centroid()) for t in tracks
        ], dtype=np.float32)

        # Euclidean distance matrix
        dist_matrix = np.sqrt(
            ((det_centroids[:, None, :] - trk_centroids[None, :, :]) ** 2).sum(-1)
        )  # shape: (N_dets, N_tracks)

        # Greedy matching — assign closest pairs first
        matched      = []
        used_dets    = set()
        used_trks    = set()

        # Flatten and sort by distance
        pairs = sorted(
            [(dist_matrix[d, t], d, t)
             for d in range(len(dets))
             for t in range(len(tracks))],
            key=lambda x: x[0]
        )

        for dist, d_idx, t_idx in pairs:
            if d_idx in used_dets or t_idx in used_trks:
                continue
            if dist > self.max_distance:
                break   # remaining pairs are all farther — stop
            matched.append((d_idx, t_idx))
            used_dets.add(d_idx)
            used_trks.add(t_idx)

        unmatched_dets = [d for d in range(len(dets)) if d not in used_dets]
        unmatched_trks = [t for t in range(len(tracks)) if t not in used_trks]

        return matched, unmatched_dets, unmatched_trks


# ── Pseudo-Label Temporal Filter ───────────────────────────
class TemporalPseudoLabelFilter:
    """
    Filters pseudo-labels using Kalman centroid tracker.

    Core idea:
        A pseudo-label is KEPT only if it belongs to a track
        that has been confirmed across min_track_hits frames.
        Single-frame detections (noise) are discarded.

    This is the key novelty for DAOD:
        Instead of accepting all high-confidence pseudo-labels,
        we require temporal consistency across frames.

    Usage:
        filter = TemporalPseudoLabelFilter()

        # Feed frames sequentially
        for frame_idx, (img_path, pseudo_labels) in enumerate(sequence):
            kept = filter.update(pseudo_labels, frame_idx)
            # kept = list of pseudo-labels with track support

        # After full sequence, get all confirmed labels
        all_confirmed = filter.get_confirmed_labels()
    """

    def __init__(self,
                 max_age      = 45,
                 min_hits     = 2,
                 max_distance = 50.0,
                 min_track_hits = 2):
        """
        min_track_hits : minimum frames a track must appear in
                         for its pseudo-labels to be accepted
        """
        self.tracker        = KalmanCentroidTracker(
            max_age      = max_age,
            min_hits     = min_hits,
            max_distance = max_distance,
        )
        self.min_track_hits = min_track_hits

        # Store confirmed pseudo-labels per frame
        # key: frame_idx, value: list of [x1,y1,x2,y2,conf,track_id]
        self.confirmed_labels = {}

        # Track hit counts — accumulated over full sequence
        self.track_hit_counts = {}

    def reset(self):
        self.tracker.reset()
        self.confirmed_labels  = {}
        self.track_hit_counts  = {}

    def update(self, pseudo_labels, frame_idx, img_path=None):
        """
        Process one frame of pseudo-labels.

        pseudo_labels: list of [x1, y1, x2, y2, conf]
                       or np.array (N, 5)
                       Empty list if no detections.

        Returns active confirmed tracks this frame.
        """
        active_tracks = self.tracker.update(pseudo_labels, frame_idx)

        # Update hit counts for each active track
        frame_labels = []
        for track in active_tracks:
            tid = track.id
            self.track_hit_counts[tid] = self.track_hit_counts.get(tid, 0) + 1

            x1, y1, x2, y2 = track.bbox
            conf = 1.0   # pseudo-label confidence (already filtered upstream)

            frame_labels.append({
                "frame_idx": frame_idx,
                "img_path":  img_path,
                "track_id":  tid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": conf,
                "track_hits": track.hits,
            })

        self.confirmed_labels[frame_idx] = frame_labels
        return active_tracks

    def get_confirmed_labels(self):
        """
        After processing full sequence, return only labels
        whose track appeared for at least min_track_hits frames.
        Discards all single-frame / low-hit pseudo-labels.
        """
        confirmed = []
        for frame_idx, labels in self.confirmed_labels.items():
            for label in labels:
                tid = label["track_id"]
                if self.track_hit_counts.get(tid, 0) >= self.min_track_hits:
                    confirmed.append(label)
        return confirmed

    def get_stats(self):
        """Summary statistics for logging."""
        total_raw     = sum(len(v) for v in self.confirmed_labels.values())
        confirmed     = self.get_confirmed_labels()
        total_tracks  = len(self.track_hit_counts)
        good_tracks   = sum(1 for h in self.track_hit_counts.values()
                            if h >= self.min_track_hits)
        return {
            "total_raw_pseudo_labels":       total_raw,
            "confirmed_pseudo_labels":       len(confirmed),
            "rejected_pseudo_labels":        total_raw - len(confirmed),
            "rejection_rate_%":              round(100*(total_raw-len(confirmed))/max(total_raw,1), 1),
            "total_tracks_created":          total_tracks,
            "confirmed_tracks":              good_tracks,
            "rejected_tracks":               total_tracks - good_tracks,
        }


# ── Quick test ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== KalmanCentroidTracker Test ===\n")

    tracker = KalmanCentroidTracker(max_age=5, min_hits=2, max_distance=50)

    # Simulate 10 frames with 2 fish moving slowly
    fish1_positions = [(100+i*2, 150+i,   130+i*2, 170+i)   for i in range(10)]
    fish2_positions = [(300-i*3, 200+i*2, 330-i*3, 220+i*2) for i in range(10)]

    # Fish 2 disappears at frame 5-6 then reappears
    for frame_idx in range(10):
        dets = [list(fish1_positions[frame_idx]) + [0.85]]
        if frame_idx not in [5, 6]:   # fish2 missing frames 5,6
            dets.append(list(fish2_positions[frame_idx]) + [0.78])

        active = tracker.update(dets, frame_idx)
        ids    = [t.id for t in active]
        print(f"Frame {frame_idx:2d} | Dets: {len(dets)} | "
              f"Active tracks: {len(active)} | IDs: {ids}")

    print(f"\nTotal unique fish IDs: {KalmanCentroidTrack.count}")

    print("\n=== TemporalPseudoLabelFilter Test ===\n")

    filter_ = TemporalPseudoLabelFilter(
        max_age=5, min_hits=2, max_distance=50, min_track_hits=3
    )

    # Simulate pseudo-labels: fish1 appears all 10 frames, noise appears only 1 frame
    for frame_idx in range(10):
        pseudo = [list(fish1_positions[frame_idx]) + [0.75]]
        if frame_idx == 3:
            pseudo.append([400, 400, 430, 420, 0.61])  # noise detection — 1 frame only
        filter_.update(pseudo, frame_idx, img_path=f"frame_{frame_idx:04d}.jpg")

    confirmed = filter_.get_confirmed_labels()
    stats     = filter_.get_stats()

    print(f"Confirmed pseudo-labels : {stats['confirmed_pseudo_labels']}")
    print(f"Rejected pseudo-labels  : {stats['rejected_pseudo_labels']}")
    print(f"Rejection rate          : {stats['rejection_rate_%']}%")
    print(f"Confirmed tracks        : {stats['confirmed_tracks']}")
    print(f"Rejected tracks         : {stats['rejected_tracks']}")
    print(f"\nSample confirmed label  : {confirmed[0] if confirmed else 'None'}")
