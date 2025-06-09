# scripts/jobs/product_detect_youtube.py
#!/usr/bin/env python
"""
Detect & track products in a YouTube video with YOLOv8 + ByteTrack.

CLI entry-point: ``product_detect_youtube``  (declared in pyproject.toml)

Example
    python product_detect_youtube.py --url https://youtu.be/8lgLYGBbDNs --model yolov8s.pt --show --save output/out.mp4

    Fine-tuned
    python product_detect_youtube.py --url https://youtu.be/8lgLYGBbDNs --model runs/detect/yolov8_brands4/weights/best.pt --show --save output/out.mp4
    
The CLI tool streams a YouTube video directly into OpenCV by letting yt-dlp supply a raw MP4/HLS URL that FFmpeg 
can read frame-by-frame; each frame is fed to a pre-loaded Ultralytics YOLOv8 detector, which 
performs convolution–BN fusion for speed, runs object detection, and immediately passes the 
results to the built-in ByteTrack module to assign persistent IDs across frames. The combined 
detection-tracking output is rendered onto the frame (boxes, class names, track IDs) and, depending on 
flags, is shown live in a pop-up window and/or written to an MP4 file via cv2.VideoWriter, 
while a Rich progress bar updates in the terminal. The entire pipeline executes locally—CPU or modest 
GPU—so inference stays on-device without downloading the video or using cloud services, giving you real-time, 
annotated video of product detections with coherent tracking identities.
"""
import argparse, sys
from yt_dlp import YoutubeDL
import cv2
from ultralytics import YOLO
from rich.progress import Progress

# ------------- argument handling --------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--url",   required=True, help="YouTube link or live URL")
    p.add_argument("--model", default="yolov8s.pt"),
                   # choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    p.add_argument("--save",  help="Write annotated video to this path")
    p.add_argument("--show",  action="store_true", help="Display live window")
    return p.parse_args()

# ------------- helper -------------------------------------------------------
def _get_stream_url(youtube_url: str) -> str:
    with YoutubeDL({"quiet": True, "format": "best[ext=mp4]/best"}) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info.get("url") or info["entries"][0]["url"]

# ------------- core ---------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(_get_stream_url(args.url))
    if not cap.isOpened():
        sys.exit("❌  Could not open stream – check URL or FFmpeg install.")

    model = YOLO(args.model)
    model.fuse()

    out = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            args.save, fourcc,
            cap.get(cv2.CAP_PROP_FPS) or 30,
            (int(cap.get(3)), int(cap.get(4))),
        )

    with Progress() as prog:
        task = prog.add_task("[cyan]Processing…",
                             total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, imgsz=640,
                                  tracker="bytetrack.yaml",
                                  persist=True, verbose=False)

            annotated = results[0].plot()

            if args.show:
                cv2.imshow("YOLOv8 ByteTrack", annotated)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

            if out:
                out.write(annotated)

            prog.advance(task)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

# ------------- entry-point for Poetry ---------------------------------------
def main() -> None:
    """Entry-point used by `poetry run product_detect_youtube`."""
    run(_parse_args())

if __name__ == "__main__":
    main()
