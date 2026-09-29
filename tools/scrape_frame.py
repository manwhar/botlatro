import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("yt-dlp is not installed. Please install it using: pip install yt-dlp")
    sys.exit(1)


def get_1080p_stream(video_url: str) -> str:
    """Extracts the direct stream URL for the 1080p video format."""
    # Force 1080p height. yt-dlp will throw an error if it doesn't exist.
    ydl_opts = {
        "quiet": True,
        "format": "bestvideo[height=1080]/best[height=1080]",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(video_url, download=False)
        except yt_dlp.utils.DownloadError as e:
            if "Requested format is not available" in str(e):
                raise ValueError("1080p format is not available for this video.") from e
            raise ValueError(f"Failed to fetch video info: {e}") from e

        if "requested_formats" in info:
            for f in info["requested_formats"]:
                if f.get("vcodec") != "none":
                    return f["url"]
        elif "url" in info:
            return info["url"]

        raise ValueError("Could not find a valid stream URL in the extracted info.")


def grab_frame(video_url: str, timestamp: str, output_path: str):
    """Grabs a specific frame using the stream URL and ffmpeg."""
    print(f"Fetching 1080p stream URL for: {video_url}")

    try:
        stream_url = get_1080p_stream(video_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting frame at {timestamp}...")

    # We use ffmpeg to quickly seek to the timestamp and dump 1 frame.
    # OpenCV's cv2.VideoCapture seeking is notoriously broken on remote HLS/DASH streams.
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite if exists
        "-ss",
        timestamp,  # Fast seek (must be before -i)
        "-i",
        stream_url,  # Input stream URL
        "-vframes",
        "1",  # Only grab 1 frame
        "-q:v",
        "2",  # High quality JPEG
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully saved frame to: {output_path}")
    except FileNotFoundError:
        print("Error: 'ffmpeg' is not installed or not in your PATH.", file=sys.stderr)
        print("ffmpeg is required to extract frames from the stream.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grab a 1080p frame from a video using yt-dlp and ffmpeg."
    )
    parser.add_argument("url", help="YouTube (or other supported) video URL")
    parser.add_argument(
        "timestamp", help="Timestamp to grab (e.g., '00:01:23.500' or '83.5')"
    )
    parser.add_argument(
        "--filename",
        "-f",
        help="Output filename (e.g., 'test_frame.jpg')",
        default="scraped_frame.jpg",
    )

    args = parser.parse_args()

    # Resolve project root relative to this script (tools/)
    script_dir = Path(__file__).parent
    out_dir = script_dir.parent / "assets" / "test_inputs" / "scraped_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / args.filename
    grab_frame(args.url, args.timestamp, str(out_path.resolve()))
