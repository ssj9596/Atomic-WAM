"""Apply the requested RoboLab video quality without modifying RoboLab."""

from __future__ import annotations

import os

import imageio.v2 as imageio


_original_get_writer = imageio.get_writer


def _get_writer(uri, *args, **kwargs):
    quality_raw = os.environ.get("ROBOLAB_VIDEO_QUALITY")
    if quality_raw is not None and str(uri).lower().endswith(".mp4"):
        quality = int(quality_raw)
        if not 0 <= quality <= 10:
            raise ValueError(f"ROBOLAB_VIDEO_QUALITY must be in [0,10], got {quality}")
        params = list(kwargs.get("ffmpeg_params") or [])
        cleaned: list[str] = []
        index = 0
        while index < len(params):
            if params[index] == "-crf":
                index += 2
                continue
            cleaned.append(params[index])
            index += 1
        kwargs["ffmpeg_params"] = cleaned
        kwargs["quality"] = quality
    return _original_get_writer(uri, *args, **kwargs)


imageio.get_writer = _get_writer
