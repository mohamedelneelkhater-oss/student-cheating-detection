"""Command line entry point: python main.py --video samples/sample_exam.mp4"""
import argparse
from src.database import AlertDatabase
from dataclasses import replace
from pathlib import Path

from src.database import AlertDatabase
from src.detector import CheatingMonitor, DetectorConfig, format_notification
from src.firebase_client import FirebaseSync


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cheating Tracking and Alert System (CLI)")
    p.add_argument("--video", required=True, help="Path to the exam video")
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--output", default="outputs/annotated.mp4", help="Annotated video path")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--min-frames", type=int, default=5, help="Consecutive frames before an alert")
    p.add_argument("--cooldown", type=float, default=10.0, help="Seconds between alerts per student")
    p.add_argument("--stride", type=int, default=1, help="Process every N-th frame")
    p.add_argument("--firebase-credentials", default=None, help="Service-account JSON (optional)")
    p.add_argument("--firebase-bucket", default=None, help="Firebase Storage bucket (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = replace(DetectorConfig(), weights=args.weights, conf=args.conf, imgsz=args.imgsz,
                  min_consecutive_frames=args.min_frames, cooldown_seconds=args.cooldown,
                  frame_stride=args.stride)
    firebase = FirebaseSync(args.firebase_credentials, args.firebase_bucket)
    monitor = CheatingMonitor(cfg, db=AlertDatabase(cfg.db_path), firebase=firebase,
                              on_alert=lambda e: print("\n" + format_notification(e)))
    summary = monitor.process_video(args.video, output_path=args.output)
    print(f"\nProcessed {summary['frames_processed']} frames in {summary['elapsed_sec']:.1f}s | "
          f"alerts: {len(summary['alerts'])} | output: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
