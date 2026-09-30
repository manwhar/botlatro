from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import cv2
from cv2.typing import MatLike

from geometry import ImageSize
from typing_extensions import Self


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    path: Path
    resolution: ImageSize
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class VideoFrame:
    image: MatLike
    index: int
    timestamp_seconds: float


class VideoReader(Iterator[VideoFrame]):
    """Iterate through a video while validating resolution and skipping frames."""

    def __init__(
        self,
        path: str | Path,
        expected_resolution: ImageSize,
        skip_frames: int = 0,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        start_timestamp_seconds: float | None = None,
        end_timestamp_seconds: float | None = None,
    ) -> None:
        if skip_frames < 0:
            raise ValueError("skip_frames must be non-negative")
        if start_frame is not None and start_timestamp_seconds is not None:
            raise ValueError("Specify either start_frame or start_timestamp_seconds")
        if end_frame is not None and end_timestamp_seconds is not None:
            raise ValueError("Specify either end_frame or end_timestamp_seconds")
        if start_frame is not None and start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if end_frame is not None and end_frame < 0:
            raise ValueError("end_frame must be non-negative")
        if start_timestamp_seconds is not None and start_timestamp_seconds < 0:
            raise ValueError("start_timestamp_seconds must be non-negative")
        if end_timestamp_seconds is not None and end_timestamp_seconds < 0:
            raise ValueError("end_timestamp_seconds must be non-negative")

        self.path = Path(path)
        self.expected_resolution = expected_resolution
        self.skip_frames = skip_frames
        self.capture = cv2.VideoCapture(str(self.path))

        if not self.capture.isOpened():
            raise ValueError(f"Could not open video: {self.path}")

        width = round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        actual_resolution = ImageSize(width=width, height=height)
        if actual_resolution != expected_resolution:
            self.close()
            raise ValueError(
                f"Video resolution is {width}x{height}; expected "
                f"{expected_resolution.width}x{expected_resolution.height}."
            )

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        frame_count = round(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.metadata = VideoMetadata(
            path=self.path,
            resolution=actual_resolution,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=frame_count / fps if fps > 0 else 0.0,
        )

        if start_timestamp_seconds is not None:
            if fps <= 0:
                self.close()
                raise ValueError(
                    "Cannot use timestamp bounds when video FPS is unavailable"
                )
            start_frame = ceil(start_timestamp_seconds * fps)
        if end_timestamp_seconds is not None:
            if fps <= 0:
                self.close()
                raise ValueError(
                    "Cannot use timestamp bounds when video FPS is unavailable"
                )
            end_frame = ceil(end_timestamp_seconds * fps)

        if start_frame is None:
            start_frame = 0
        if end_frame is not None and end_frame < start_frame:
            self.close()
            raise ValueError("end_frame must be greater than or equal to start_frame")

        self._end_frame = end_frame
        self._next_index = start_frame
        if start_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def __iter__(self) -> "VideoReader":
        return self

    def __next__(self) -> VideoFrame:
        if self._end_frame is not None and self._next_index >= self._end_frame:
            self.close()
            raise StopIteration

        success, image = self.capture.read()
        if not success:
            self.close()
            raise StopIteration

        frame_index = self._next_index
        timestamp_seconds = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000

        self._next_index += 1

        for _ in range(self.skip_frames):
            if not self.capture.grab():
                break
            self._next_index += 1

        return VideoFrame(
            image=image,
            index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )

    def close(self) -> None:
        if self.capture.isOpened():
            self.capture.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def iter_video_frames(
    path: str | Path,
    expected_resolution: ImageSize,
    skip_frames: int = 0,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_timestamp_seconds: float | None = None,
    end_timestamp_seconds: float | None = None,
) -> Iterator[VideoFrame]:
    """Yield a bounded range of video frames and always release the capture.

    Frame endpoints are zero-based and the end endpoint is exclusive. Timestamp
    endpoints are measured in seconds and are converted to the first frame at
    or after each timestamp.
    """
    with VideoReader(
        path,
        expected_resolution,
        skip_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        start_timestamp_seconds=start_timestamp_seconds,
        end_timestamp_seconds=end_timestamp_seconds,
    ) as reader:
        yield from reader
