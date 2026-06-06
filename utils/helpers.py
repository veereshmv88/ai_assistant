"""
utils/helpers.py — Shared Utility Functions
=============================================
Common utilities used across modules:
  • Image encoding (frame → base64 JPEG)
  • Frame resizing
  • Bearing conversion to human words
  • Distance formatting
  • Safe async sleep with cancellation
"""

import asyncio
import base64
import math
from typing import Optional

import cv2
import numpy as np


def frame_to_base64(frame: np.ndarray, quality: int = 70) -> str:
    """
    Encode an OpenCV frame (BGR numpy array) as a base64 JPEG string.
    Used for sending images to Ollama and other APIs.

    Args:
        frame:   BGR numpy array from OpenCV
        quality: JPEG quality 1-100 (lower = smaller = faster)

    Returns:
        base64-encoded string
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def frame_to_bytes(frame: np.ndarray, quality: int = 70) -> bytes:
    """Encode frame as raw JPEG bytes."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()


def resize_frame(
    frame: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_side: Optional[int] = None,
) -> np.ndarray:
    """
    Resize frame while preserving aspect ratio.

    Args:
        frame:    Input frame
        width:    Target width (height auto-calculated)
        height:   Target height (width auto-calculated)
        max_side: Limit longest side to this value

    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]

    if max_side:
        if w > h:
            width = max_side
        else:
            height = max_side

    if width and not height:
        ratio = width / w
        height = int(h * ratio)
    elif height and not width:
        ratio = height / h
        width = int(w * ratio)
    elif not width and not height:
        return frame

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def bearing_to_words(bearing: float) -> str:
    """
    Convert compass bearing (0–360°) to cardinal direction words.

    Args:
        bearing: Compass heading in degrees (0 = North)

    Returns:
        Human-readable direction string
    """
    directions = [
        (0,   "north"),
        (22,  "north-north-east"),
        (45,  "north-east"),
        (67,  "east-north-east"),
        (90,  "east"),
        (112, "east-south-east"),
        (135, "south-east"),
        (157, "south-south-east"),
        (180, "south"),
        (202, "south-south-west"),
        (225, "south-west"),
        (247, "west-south-west"),
        (270, "west"),
        (292, "west-north-west"),
        (315, "north-west"),
        (337, "north-north-west"),
        (360, "north"),
    ]
    bearing = bearing % 360
    for threshold, label in reversed(directions):
        if bearing >= threshold:
            return label
    return "north"


def format_distance(metres: float) -> str:
    """Format distance in human-friendly speech."""
    if metres < 1:
        return "less than one metre"
    elif metres < 10:
        return f"{int(metres)} metres"
    elif metres < 100:
        return f"{int(round(metres / 5) * 5)} metres"
    elif metres < 1000:
        return f"{int(round(metres / 10) * 10)} metres"
    else:
        km = metres / 1000
        return f"{km:.1f} kilometres"


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value within [min_val, max_val]."""
    return max(min_val, min(max_val, value))


async def safe_sleep(seconds: float):
    """Sleep that catches CancelledError gracefully."""
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        pass


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance in metres between two GPS points.
    """
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def is_daylight(hour: int = None) -> bool:
    """Simple heuristic: daylight between 6am and 8pm."""
    import datetime
    if hour is None:
        hour = datetime.datetime.now().hour
    return 6 <= hour < 20
