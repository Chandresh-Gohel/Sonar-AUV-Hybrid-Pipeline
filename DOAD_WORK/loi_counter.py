"""
=============================================================
  Line of Interest (LOI) Fish Counter

  Counts fish crossing a VERTICAL line in the center of frame.
  Fish are counted LEFT or RIGHT based on trajectory direction.

  This matches the exact Caltech CFC counting protocol:
  "A vertical LOI is drawn in the middle of the frame.
   A fish is counted left or right if its trajectory start
   and end positions are on different sides of the LOI."
   — Kay et al., ECCV 2022

  Key rule: only fish whose trajectory STARTS and ENDS on
  opposite sides are counted. Fish that enter and exit the
  same side are excluded (matches field technician protocol).

  ARIS sonar setup:
    Fish swim perpendicular to camera beam (left-right)
    Right = upstream (salmon migrating to spawn)
    Left  = downstream

  Usage:
    counter = LOICounter(frame_width=512)
    for frame:
        counter.update_sort(tracks)   # or update_kc(active)
    print(counter.get_counts())
=============================================================
"""


class LOICounter:
    """
    Counts fish crossing a vertical line at line_ratio * frame_width.

    Counts:
      count_right : fish moving left → right (upstream)
      count_left  : fish moving right → left (downstream)
      total       : sum of both directions

    Only counts fish whose trajectory spans both sides of the line
    (start on one side, end on the other). Fish that never cross
    are excluded — matches CFC field technician protocol.
    """

    def __init__(self, frame_width, line_ratio=0.5):
        """
        frame_width : width of the original (not letterboxed) frame
        line_ratio  : position of vertical LOI (0.5 = center)
        """
        self.line_x     = int(frame_width * line_ratio)
        self.line_ratio = line_ratio

        # Per-track state
        self.track_start_x  = {}   # track_id -> x at first detection
        self.track_current_x = {}  # track_id -> current x centroid
        self.counted_ids    = set() # IDs already counted (avoid double)

        self.count_right = 0   # left → right crossings
        self.count_left  = 0   # right → left crossings

    def _cx(self, bbox):
        """X centroid from [x1,y1,x2,y2]."""
        return (bbox[0] + bbox[2]) / 2.0

    def update_sort(self, tracks):
        """
        Update with SORT tracks.
        tracks: numpy array (N,5) [x1,y1,x2,y2,track_id]
        """
        active_ids = set()
        for t in tracks:
            x1, y1, x2, y2, tid = t
            tid = int(tid)
            cx  = (x1 + x2) / 2.0
            active_ids.add(tid)
            self._update_track(tid, cx)

        # Check completed tracks (no longer active)
        self._finalize_lost(active_ids)

    def update_kc(self, active_tracks):
        """
        Update with Kalman centroid tracks.
        active_tracks: list of track objects with .id and .bbox
        """
        active_ids = set()
        for t in active_tracks:
            cx = self._cx(t.bbox)
            active_ids.add(t.id)
            self._update_track(t.id, cx)

        self._finalize_lost(active_ids)

    def _update_track(self, tid, cx):
        """Record start and current x for a track."""
        if tid not in self.track_start_x:
            self.track_start_x[tid] = cx
        self.track_current_x[tid] = cx

    def _finalize_lost(self, active_ids):
        """
        When a track disappears, check if it crossed the line.
        Count only if start and end are on opposite sides.
        """
        lost_ids = set(self.track_current_x.keys()) - active_ids
        for tid in lost_ids:
            if tid not in self.counted_ids:
                start_x   = self.track_start_x.get(tid)
                current_x = self.track_current_x.get(tid)
                if start_x is not None and current_x is not None:
                    self._check_crossing(tid, start_x, current_x)
            # Clean up
            self.track_start_x.pop(tid, None)
            self.track_current_x.pop(tid, None)

    def _check_crossing(self, tid, start_x, end_x):
        """Count if trajectory crosses LOI (start and end on opposite sides)."""
        if tid in self.counted_ids:
            return
        start_side = start_x < self.line_x
        end_side   = end_x   < self.line_x
        if start_side != end_side:
            # Crossed the line
            if start_x < end_x:
                self.count_right += 1   # moved left → right
            else:
                self.count_left += 1    # moved right → left
            self.counted_ids.add(tid)

    def finalize(self):
        """
        Call at end of video to count any still-active tracks
        that crossed the line but haven't been finalized yet.
        """
        all_ids = set(self.track_current_x.keys())
        for tid in all_ids:
            if tid not in self.counted_ids:
                start_x   = self.track_start_x.get(tid)
                current_x = self.track_current_x.get(tid)
                if start_x is not None and current_x is not None:
                    self._check_crossing(tid, start_x, current_x)

    def get_counts(self):
        return {
            "count_right": self.count_right,   # upstream
            "count_left":  self.count_left,    # downstream
            "total":       self.count_right + self.count_left,
        }

    def draw_loi(self, frame, scale=1.0, pad_x=0, pad_y=0):
        """
        Draw vertical LOI line and counts on letterboxed frame.
        """
        import cv2
        h, w = frame.shape[:2]

        # Scale LOI to letterboxed coordinates
        lx = int(self.line_x * scale + pad_x)

        # Draw vertical line
        cv2.line(frame, (lx, 0), (lx, h), (0, 255, 255), 2)
        cv2.putText(frame, "LOI", (lx + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw counts
        counts = self.get_counts()
        cv2.putText(frame, f"R:{counts['count_right']}",
                    (lx + 5, h // 2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame, f"L:{counts['count_left']}",
                    (lx + 5, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        return frame
