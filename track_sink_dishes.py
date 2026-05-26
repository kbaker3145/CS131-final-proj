#!/usr/bin/env python3
"""
Track kitchen items in a sink ROI from a prerecorded video.

Outputs JSON with add/remove events (Option A) and per-track summaries (Option C).
Face recognition is separate; merge later by matching timestamps.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:
    print(
        "Missing OpenCV. Install project dependencies:\n"
        "  cd CS131-final-proj && python3 -m venv .venv && source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "The pip package is opencv-python (import name is cv2), not 'cv2'.",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

from ultralytics import YOLO

# COCO proxies for dishes (no "plate" in COCO)
DEFAULT_KITCHEN_CLASSES = frozenset(
    {"bowl", "cup", "fork", "knife", "spoon", "bottle", "wine glass"}
)


@dataclass
class Roi:
    x: int
    y: int
    w: int
    h: int

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def contains_bbox_center(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return self.contains_point(cx, cy)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_string(cls, s: str) -> Roi:
        parts = [int(p.strip()) for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError("ROI must be x,y,w,h")
        x, y, w, h = parts
        if w <= 0 or h <= 0:
            raise ValueError("ROI width and height must be positive")
        return cls(x=x, y=y, w=w, h=h)

    @classmethod
    def from_json_file(cls, path: Path) -> Roi:
        data = json.loads(path.read_text())
        return cls(x=int(data["x"]), y=int(data["y"]), w=int(data["w"]), h=int(data["h"]))


@dataclass
class TrackState:
    class_name: str
    first_seen_frame: int
    last_seen_frame: int
    added_frame: int | None = None
    removed_frame: int | None = None
    status: str = "tentative"  # tentative | in_sink | removed | fixture
    consecutive_present: int = 0
    consecutive_absent: int = 0


def parse_kitchen_classes(s: str | None) -> frozenset[str]:
    if not s:
        return DEFAULT_KITCHEN_CLASSES
    return frozenset(c.strip().lower() for c in s.split(",") if c.strip())


def load_first_frame(video_path: Path) -> Any:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame from: {video_path}")
    return frame


def pick_roi_interactive(frame: Any, window_title: str = "Draw sink ROI, press ENTER") -> Roi:
    print(window_title)
    print("Drag a rectangle over the sink basin, then press ENTER or SPACE to confirm.")
    x, y, w, h = cv2.selectROI(window_title, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        raise RuntimeError("No ROI selected.")
    return Roi(x=int(x), y=int(y), w=int(w), h=int(h))


def frame_to_sec(frame_idx: int, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return round(frame_idx / fps, 3)


def run_tracking(
    video_path: Path,
    roi: Roi,
    output_path: Path,
    *,
    model_name: str = "yolov8s.pt",
    conf: float = 0.35,
    iou: float = 0.45,
    kitchen_classes: frozenset[str],
    min_present_frames: int = 8,
    min_absent_frames: int = 15,
    baseline_sec: float = 2.0,
    frame_stride: int = 1,
    max_frames: int | None = None,
    save_roi_path: Path | None = None,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if save_roi_path:
        save_roi_path.write_text(json.dumps(roi.to_dict(), indent=2) + "\n")

    model = YOLO(model_name)
    names = model.names  # id -> class name

    tracks: dict[int, TrackState] = {}
    events: list[dict[str, Any]] = []
    in_sink: set[int] = set()
    fixture_ids: set[int] = set()
    baseline_cutoff_frame = int(baseline_sec * fps)

    frames_processed = 0
    source_frame = -frame_stride  # first iteration lands at 0

    def emit_event(event_type: str, track_id: int, ts: TrackState) -> None:
        events.append(
            {
                "type": event_type,
                "track_id": track_id,
                "class": ts.class_name,
                "time_sec": frame_to_sec(
                    ts.added_frame if event_type == "added" else ts.removed_frame or 0,
                    fps,
                ),
                "frame": ts.added_frame if event_type == "added" else ts.removed_frame,
            }
        )

    stream = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        conf=conf,
        iou=iou,
        verbose=False,
        tracker="bytetrack.yaml",
        vid_stride=max(1, frame_stride),
    )

    for result in stream:
        if max_frames is not None and frames_processed >= max_frames:
            break
        source_frame += frame_stride
        frames_processed += 1
        frame_idx = source_frame  # index in original video timeline

        if result.boxes is None or len(result.boxes) == 0:
            active_ids: set[int] = set()
        else:
            active_ids = set()
            boxes = result.boxes
            for i in range(len(boxes)):
                if boxes.id is None:
                    continue
                track_id = int(boxes.id[i].item())
                cls_id = int(boxes.cls[i].item())
                class_name = names[cls_id].lower()
                if class_name not in kitchen_classes:
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                if not roi.contains_bbox_center(x1, y1, x2, y2):
                    continue

                active_ids.add(track_id)

                if track_id not in tracks:
                    tracks[track_id] = TrackState(
                        class_name=class_name,
                        first_seen_frame=frame_idx,
                        last_seen_frame=frame_idx,
                    )
                ts = tracks[track_id]
                ts.last_seen_frame = frame_idx
                if ts.class_name != class_name:
                    # Keep first label; appearance can confuse COCO across frames
                    pass

                ts.consecutive_present += 1
                ts.consecutive_absent = 0

                if (
                    track_id not in in_sink
                    and track_id not in fixture_ids
                    and ts.consecutive_present >= min_present_frames
                ):
                    if frame_idx <= baseline_cutoff_frame:
                        fixture_ids.add(track_id)
                        ts.status = "fixture"
                    else:
                        in_sink.add(track_id)
                        ts.status = "in_sink"
                        ts.added_frame = frame_idx
                        emit_event("added", track_id, ts)

        # Tracks that were in sink but not seen this frame
        for track_id in list(in_sink):
            if track_id in active_ids:
                continue
            ts = tracks[track_id]
            ts.consecutive_absent += 1
            ts.consecutive_present = 0
            if ts.consecutive_absent >= min_absent_frames:
                in_sink.discard(track_id)
                ts.status = "removed"
                ts.removed_frame = frame_idx
                emit_event("removed", track_id, ts)

        # Tentative tracks outside in_sink that disappeared before confirmation
        for track_id, ts in list(tracks.items()):
            if track_id in in_sink or track_id in fixture_ids:
                continue
            if track_id in active_ids:
                continue
            ts.consecutive_absent += 1

    # Mark still-present dishes at end of clip
    for track_id in in_sink:
        tracks[track_id].status = "in_sink"

    track_summaries: dict[str, dict[str, Any]] = {}
    for track_id, ts in sorted(tracks.items()):
        if ts.status == "tentative" and track_id not in fixture_ids:
            continue
        summary: dict[str, Any] = {
            "class": ts.class_name,
            "status": ts.status,
            "first_seen_sec": frame_to_sec(ts.first_seen_frame, fps),
            "last_seen_sec": frame_to_sec(ts.last_seen_frame, fps),
            "first_seen_frame": ts.first_seen_frame,
            "last_seen_frame": ts.last_seen_frame,
        }
        if ts.added_frame is not None:
            summary["added_sec"] = frame_to_sec(ts.added_frame, fps)
            summary["added_frame"] = ts.added_frame
        if ts.removed_frame is not None:
            summary["removed_sec"] = frame_to_sec(ts.removed_frame, fps)
            summary["removed_frame"] = ts.removed_frame
        track_summaries[str(track_id)] = summary

    output: dict[str, Any] = {
        "video": str(video_path.resolve()),
        "fps": round(fps, 3),
        "frame_count_reported": total_frames_reported,
        "frames_processed": frames_processed,
        "frame_stride": frame_stride,
        "roi": roi.to_dict(),
        "model": model_name,
        "conf": conf,
        "kitchen_classes": sorted(kitchen_classes),
        "parameters": {
            "min_present_frames": min_present_frames,
            "min_absent_frames": min_absent_frames,
            "baseline_sec": baseline_sec,
        },
        "events": events,
        "tracks": track_summaries,
        "summary": {
            "add_count": sum(1 for e in events if e["type"] == "added"),
            "remove_count": sum(1 for e in events if e["type"] == "removed"),
            "fixtures_ignored": len(fixture_ids),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Track dishes in a sink ROI and write add/remove events to JSON.",
    )
    p.add_argument("--video", "-v", type=Path, required=True, help="Input video path")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output/sink_events.json"),
        help="Output JSON path",
    )
    roi = p.add_mutually_exclusive_group(required=True)
    roi.add_argument("--roi", type=str, help="Sink region as x,y,w,h (pixels)")
    roi.add_argument("--roi-file", type=Path, help="JSON file with x,y,w,h")
    roi.add_argument(
        "--pick-roi",
        action="store_true",
        help="Interactively select ROI on the first video frame",
    )
    p.add_argument(
        "--pick-roi-from-image",
        type=Path,
        help="Select ROI on a still image (e.g. setup screenshot) instead of video",
    )
    p.add_argument(
        "--save-roi",
        type=Path,
        help="Write chosen ROI to JSON for reuse (--roi-file next run)",
    )
    p.add_argument("--model", default="yolov8s.pt", help="Ultralytics weights")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    p.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated COCO class names (default: kitchen set)",
    )
    p.add_argument(
        "--min-present-frames",
        type=int,
        default=8,
        help="Frames visible before counting as added",
    )
    p.add_argument(
        "--min-absent-frames",
        type=int,
        default=15,
        help="Frames missing before counting as removed",
    )
    p.add_argument(
        "--baseline-sec",
        type=float,
        default=2.0,
        help="Tracks stable in the first N seconds are treated as fixtures (sponge, etc.)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth source frame (timestamps still use real frame numbers)",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames")
    return p


def resolve_roi(args: argparse.Namespace) -> Roi:
    if args.roi:
        return Roi.from_string(args.roi)
    if args.roi_file:
        return Roi.from_json_file(args.roi_file)
    if args.pick_roi_from_image:
        frame = cv2.imread(str(args.pick_roi_from_image))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {args.pick_roi_from_image}")
        return pick_roi_interactive(frame, "Select sink ROI on image")
    if args.pick_roi:
        frame = load_first_frame(args.video)
        return pick_roi_interactive(frame, "Select sink ROI on first video frame")
    raise RuntimeError("ROI required")


def main() -> int:
    args = build_parser().parse_args()
    if not args.video.is_file():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1

    try:
        roi = resolve_roi(args)
    except (ValueError, RuntimeError) as e:
        print(f"ROI error: {e}", file=sys.stderr)
        return 1

    kitchen = parse_kitchen_classes(args.classes)
    print(f"ROI: {roi.to_dict()}")
    print(f"Kitchen classes: {', '.join(sorted(kitchen))}")
    print("Running tracker (first run may download YOLO weights)...")

    try:
        result = run_tracking(
            args.video,
            roi,
            args.output,
            model_name=args.model,
            conf=args.conf,
            iou=args.iou,
            kitchen_classes=kitchen,
            min_present_frames=args.min_present_frames,
            min_absent_frames=args.min_absent_frames,
            baseline_sec=args.baseline_sec,
            frame_stride=max(1, args.stride),
            max_frames=args.max_frames,
            save_roi_path=args.save_roi,
        )
    except Exception as e:
        print(f"Tracking failed: {e}", file=sys.stderr)
        return 1

    s = result["summary"]
    print(f"Wrote {args.output}")
    print(f"  Events: {s['add_count']} added, {s['remove_count']} removed")
    print(f"  Fixtures ignored (baseline): {s['fixtures_ignored']}")
    print(f"  Tracks in summary: {len(result['tracks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
