"""Headless HTTP plugin server for Darask Paint.

Exposes a minimal, stable JSON/binary API on 127.0.0.1 that Darask Paint's
Rust core talks to (see docs/SPEC.md section 55/56 in the darask-paint
repository). It drives a locally managed ComfyUI instance to run image
generation ("AI generate") and masked inpainting ("AI replace").

Design notes (see the implementation report for the full rationale):

* This file uses only the Python standard library. It intentionally does
  NOT import anything from the ``ai_diffusion`` package (this fork's Krita
  plugin code) -- not even the vendored ``websockets`` submodule, which is
  therefore not required to build or run this server. That package's
  ``backend`` modules (client.py, comfy_client.py, workflow.py, server.py)
  look Qt-independent from their module names, but statically importing
  them pulls in PyQt5 for real functionality: networking is built on
  ``QNetworkAccessManager`` (ai_diffusion/backend/network.py), the async
  plumbing (ai_diffusion/eventloop.py) is a QTimer-driven pump for a
  *second*, Qt-owned event loop, and the settings/style/model layers are
  QObject subclasses. None of that touches the real ``krita`` module
  (verified: it is only ever imported behind
  ``importlib.util.find_spec("krita")`` guards), but it does require a
  running Qt event loop to do any actual I/O. That conflicts with this
  file's requirement to drive its own asyncio-free, Qt-free event loop, so
  this server talks to ComfyUI's plain HTTP API directly with hand-built
  workflow graphs instead of reusing comfy_client.py/workflow.py.
* ``ai_diffusion/backend/resources.py`` (ComfyUI version pin, default
  checkpoint list) *is* pure stdlib and was read (not imported at runtime)
  to keep the versions used by darask-plugin.bat in sync with upstream.
* This server assumes it owns a *dedicated* ComfyUI instance (one per
  darask-plugin.bat launch, matching the fork's own single-user model). It
  is not safe to point --comfy-url at a ComfyUI instance shared with other
  clients: job cancellation on timeout uses ComfyUI's global /interrupt
  (which stops whatever is currently executing, not a specific prompt id)
  because this ComfyUI version has no per-prompt interrupt API. See
  ComfyClient.cancel() and README_darask.md.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import binascii
import io
import json
import logging
import math
import random
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

# Optional acceleration for the PNG codec below. Pillow ships with every
# ComfyUI environment (darask-plugin.bat runs this file with ComfyUI's venv
# python), but it is *not* required: the pure-stdlib implementation is the
# reference and remains the fallback so the server still runs on a bare
# Python install (as do the unit tests).
try:
    from PIL import Image as _PILImage
except ImportError:  # pragma: no cover - depends on the environment
    _PILImage = None

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DARASK_PLUGIN_NAME = "darask-ai-diffusion"
DARASK_PLUGIN_VERSION = "1.52.1"  # tracks the Acly/krita-ai-diffusion fork version this is based on
API_VERSION = 1

DEFAULT_PORT = 8424
DEFAULT_HOST = (
    "127.0.0.1"  # never configurable: darask-paint only ever talks to localhost (spec 55.1)
)

MAX_IMAGE_DIM = 8192
MAX_TOTAL_PIXELS = MAX_IMAGE_DIM * MAX_IMAGE_DIM
MAX_DECODED_IMAGE_BYTES = 64 * 1024 * 1024  # 64 MiB per decoded (client-supplied) PNG
MAX_BASE64_FIELD_LEN = (MAX_DECODED_IMAGE_BYTES * 4 // 3) + 1024
MAX_REQUEST_BODY_BYTES = 2 * MAX_BASE64_FIELD_LEN + 64 * 1024  # image + mask + json overhead
MAX_PROMPT_LEN = 4000

DEFAULT_GENERATION_TIMEOUT = 300.0  # seconds, wall clock budget for one generate/inpaint call
COMFY_STARTUP_GRACE = 180.0  # seconds we report "starting" instead of "error" while Comfy boots
COMFY_POLL_INTERVAL = 0.5
COMFY_JSON_MAX_BYTES = (
    16 * 1024 * 1024
)  # cap on JSON responses from ComfyUI (/prompt, /history, ...)
COMFY_IMAGE_MAX_BYTES = 128 * 1024 * 1024  # cap on binary image responses from ComfyUI (/view)
COMFY_DEFAULT_OP_TIMEOUT = 10.0

LOCK_WAIT_TIMEOUT = (
    1.0  # fail fast on busy; single-flight is darask-paint's responsibility (spec 55.1)
)
SOCKET_READ_TIMEOUT = 30.0  # Slowloris defense: per-recv timeout on accepted connections
DEFAULT_MAX_CONCURRENT_HANDLERS = 8  # thread-exhaustion defense

CLEANUP_MAX_AGE_SECONDS = 3600.0  # startup sweep: remove leftover darask_* files older than this

DEFAULT_STEPS = 20
DEFAULT_CFG = 7.0
DEFAULT_SAMPLER = "dpmpp_2m"
DEFAULT_SCHEDULER = "karras"
DEFAULT_INPAINT_GROW_MASK = 6

log = logging.getLogger("darask_server")


# --------------------------------------------------------------------------
# Strict JSON parsing (reject NaN/Infinity, which the stdlib json module
# accepts by default even though they are not valid JSON)
# --------------------------------------------------------------------------


def _reject_json_constant(token: str) -> NoReturn:
    raise ValueError(f"Invalid JSON constant: {token}")


def strict_json_loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_json_constant)


# --------------------------------------------------------------------------
# Minimal pure-stdlib PNG decode/encode/pad/crop
#
# Supports only what this server needs to receive from a Rust image encoder
# and from ComfyUI's SaveImage node: 8-bit depth, non-interlaced, color
# types 0 (gray), 2 (RGB), 4 (gray+alpha), 6 (RGBA). Palette images (color
# type 3) and interlaced images are rejected with a clear error.
#
# Header validation (signature/IHDR/bit depth/color type/interlace) is always
# done by the stdlib code so the accepted input set does not depend on
# whether Pillow is present. Only the pixel work (un-filtering on decode,
# filtering + deflate on encode) is delegated to Pillow when it is importable:
# the pure-Python per-byte un-filter loop costs ~1.3 s for a 1024x1024 RGBA
# photo (ComfyUI output) versus ~0.03 s in Pillow, and that cost is paid on
# every generate/inpaint whose size is not a multiple of 8.
# --------------------------------------------------------------------------

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}
_PNG_PIL_MODES = {0: "L", 2: "RGB", 4: "LA", 6: "RGBA"}


class PngError(Exception):
    pass


def png_dimensions(data: bytes) -> tuple[int, int]:
    """Read width/height straight from the PNG IHDR chunk without decoding pixels."""
    if len(data) < 24 or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise PngError("Not a valid PNG image")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _png_read_chunks(data: bytes) -> list[tuple[bytes, memoryview]]:
    if len(data) < 8 or data[:8] != PNG_SIGNATURE:
        raise PngError("Not a valid PNG file (bad signature)")
    pos = 8
    chunks = []
    view = memoryview(data)
    n = len(data)
    while pos + 8 <= n:
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        tag = data[pos + 4 : pos + 8]
        start = pos + 8
        end = start + length
        if end + 4 > n:
            raise PngError("Truncated PNG chunk")
        chunks.append((tag, view[start:end]))
        pos = end + 4  # skip CRC
        if tag == b"IEND":
            break
    if not chunks or chunks[0][0] != b"IHDR":
        raise PngError("PNG missing IHDR as first chunk")
    return chunks


def png_decode(data: bytes) -> tuple[int, int, int, bytearray]:
    """Decode a PNG into (width, height, color_type, raw_pixels).

    raw_pixels is a flat bytearray of `height` scanlines of `width * channels`
    bytes each (channels depend on color_type), already un-filtered.
    """
    chunks = _png_read_chunks(data)
    _tag, ihdr = chunks[0]
    if len(ihdr) < 13:
        raise PngError("Malformed IHDR")
    width, height, bit_depth, color_type, _comp, _filt, interlace = struct.unpack(
        ">IIBBBBB", ihdr[:13]
    )
    if width <= 0 or height <= 0:
        raise PngError("Invalid PNG dimensions")
    if bit_depth != 8:
        raise PngError(f"Unsupported PNG bit depth: {bit_depth} (only 8-bit is supported)")
    if color_type not in _PNG_CHANNELS:
        raise PngError(
            f"Unsupported PNG color type: {color_type} (palette images are not supported)"
        )
    if interlace != 0:
        raise PngError("Interlaced PNG images are not supported")

    idat: list[memoryview] = []
    for tag, payload in chunks:
        if tag == b"IDAT":
            idat.append(payload)
    if not any(idat):
        raise PngError("PNG has no image data")

    fast = _png_decode_pixels_pil(data, width, height, color_type)
    if fast is not None:
        return width, height, color_type, fast

    try:
        raw = zlib.decompress(idat[0] if len(idat) == 1 else b"".join(idat))
    except zlib.error as e:
        raise PngError(f"Failed to inflate PNG image data: {e}") from e

    channels = _PNG_CHANNELS[color_type]
    bpp = channels
    stride = width * channels
    if len(raw) < (stride + 1) * height:
        raise PngError("PNG image data shorter than expected")

    out = bytearray(stride * height)
    prev_row = bytearray(stride)
    raw_view = memoryview(raw)
    pos = 0
    for row in range(height):
        filter_type = raw[pos]
        pos += 1
        cur = bytearray(raw_view[pos : pos + stride])
        pos += stride
        _png_unfilter_row(filter_type, cur, prev_row, bpp)
        out[row * stride : (row + 1) * stride] = cur
        prev_row = cur
    return width, height, color_type, out


def _png_decode_pixels_pil(
    data: bytes, width: int, height: int, color_type: int
) -> bytearray | None:
    """Un-filter the scanlines of an already header-validated PNG with Pillow.

    Returns None (caller falls back to the stdlib path, which then reports the
    precise error) when Pillow is unavailable, cannot decode the data, or
    yields anything other than the exact 8-bit mode/size implied by the IHDR.
    """
    if _PILImage is None:
        return None
    mode = _PNG_PIL_MODES.get(color_type)
    if mode is None:
        return None
    try:
        with _PILImage.open(io.BytesIO(data), formats=["PNG"]) as img:
            if img.mode != mode or img.size != (width, height):
                return None
            pixels = img.tobytes()
    except Exception:
        return None
    if len(pixels) != width * height * _PNG_CHANNELS[color_type]:
        return None
    return bytearray(pixels)


def _png_unfilter_row(filter_type: int, cur: bytearray, prev: bytearray, bpp: int) -> None:
    n = len(cur)
    if filter_type == 0:
        return
    elif filter_type == 1:  # Sub
        for i in range(bpp, n):
            cur[i] = (cur[i] + cur[i - bpp]) & 0xFF
    elif filter_type == 2:  # Up
        for i in range(n):
            cur[i] = (cur[i] + prev[i]) & 0xFF
    elif filter_type == 3:  # Average
        for i in range(n):
            a = cur[i - bpp] if i >= bpp else 0
            cur[i] = (cur[i] + ((a + prev[i]) >> 1)) & 0xFF
    elif filter_type == 4:  # Paeth
        for i in range(n):
            a = cur[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            cur[i] = (cur[i] + pr) & 0xFF
    else:
        raise PngError(f"Unsupported PNG filter type: {filter_type}")


def png_encode(width: int, height: int, color_type: int, raw: bytes | bytearray) -> bytes:
    """Encode raw (unfiltered) scanline pixel bytes into a PNG using filter type 0 (None)."""
    if color_type not in _PNG_CHANNELS:
        raise PngError(f"Unsupported color type for encoding: {color_type}")
    channels = _PNG_CHANNELS[color_type]
    stride = width * channels
    if len(raw) != stride * height:
        raise PngError("Raw pixel buffer size does not match width/height/color_type")

    fast = _png_encode_pil(width, height, color_type, raw)
    if fast is not None:
        return fast

    filtered = bytearray((stride + 1) * height)
    for row in range(height):
        src_off = row * stride
        dst_off = row * (stride + 1)
        filtered[dst_off] = 0  # filter type: None
        filtered[dst_off + 1 : dst_off + 1 + stride] = raw[src_off : src_off + stride]

    compressed = zlib.compress(filtered, 6)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def _png_encode_pil(
    width: int, height: int, color_type: int, raw: bytes | bytearray
) -> bytes | None:
    """Encode validated raw scanlines with Pillow (8-bit, non-interlaced, same
    color type as the stdlib encoder). Returns None to fall back to the stdlib
    encoder when Pillow is unavailable or fails."""
    if _PILImage is None:
        return None
    mode = _PNG_PIL_MODES.get(color_type)
    if mode is None:
        return None
    try:
        img = _PILImage.frombuffer(mode, (width, height), bytes(raw), "raw", mode, 0, 1)
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=6)
    except Exception:
        return None
    return buf.getvalue()


def png_pad_to(data: bytes, target_w: int, target_h: int) -> bytes:
    """Decode, edge-extend-pad (right/bottom only) to target size, re-encode."""
    width, height, color_type, raw = png_decode(data)
    if target_w < width or target_h < height:
        raise PngError("Cannot pad to a smaller size")
    if target_w == width and target_h == height:
        return png_encode(width, height, color_type, raw)

    channels = _PNG_CHANNELS[color_type]
    src_stride = width * channels
    dst_stride = target_w * channels
    out = bytearray(dst_stride * target_h)
    for row in range(height):
        src_off = row * src_stride
        dst_off = row * dst_stride
        out[dst_off : dst_off + src_stride] = raw[src_off : src_off + src_stride]
        if target_w > width:
            last_pixel = bytes(raw[src_off + src_stride - channels : src_off + src_stride])
            out[dst_off + src_stride : dst_off + dst_stride] = last_pixel * (target_w - width)
    if target_h > height:
        last_row = out[(height - 1) * dst_stride : height * dst_stride]
        for row in range(height, target_h):
            out[row * dst_stride : (row + 1) * dst_stride] = last_row
    return png_encode(target_w, target_h, color_type, out)


def png_crop_to(data: bytes, target_w: int, target_h: int) -> bytes:
    """Decode and crop the top-left target_w x target_h region, re-encode."""
    width, height, color_type, raw = png_decode(data)
    if target_w > width or target_h > height:
        raise PngError("Cannot crop to a larger size")
    if target_w == width and target_h == height:
        return png_encode(width, height, color_type, raw)

    channels = _PNG_CHANNELS[color_type]
    src_stride = width * channels
    dst_stride = target_w * channels
    out = bytearray(dst_stride * target_h)
    for row in range(target_h):
        src_off = row * src_stride
        dst_off = row * dst_stride
        out[dst_off : dst_off + dst_stride] = raw[src_off : src_off + dst_stride]
    return png_encode(target_w, target_h, color_type, out)


def next_multiple_of_8(value: int) -> int:
    return (value + 7) // 8 * 8


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


class ApiError(Exception):
    """Raised by request handling code; carries an HTTP status and a message."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def decode_base64_image(field_name: str, value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{field_name}' must be a non-empty base64 string")
    if len(value) > MAX_BASE64_FIELD_LEN:
        raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"'{field_name}' is too large")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{field_name}' is not valid base64") from None
    if len(data) > MAX_DECODED_IMAGE_BYTES:
        raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"'{field_name}' is too large")
    return data


def validate_extent(width: int, height: int) -> None:
    """Validate a requested/actual image extent. Arbitrary 1..8192 sizes are
    accepted (no multiple-of-8 requirement) -- the server internally pads to
    a multiple of 8 for ComfyUI and crops the result back (see pad/crop
    handling in _handle_generate/_handle_inpaint)."""
    for name, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' must be an integer")
        if value <= 0 or value > MAX_IMAGE_DIM:
            raise ApiError(
                HTTPStatus.BAD_REQUEST, f"'{name}' must be between 1 and {MAX_IMAGE_DIM}"
            )
    if width * height > MAX_TOTAL_PIXELS:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"width*height must not exceed {MAX_TOTAL_PIXELS}")


def validate_padded_extent(width: int, height: int) -> None:
    """Defense in depth: re-validate dimensions after padding to a multiple of 8."""
    if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM or width * height > MAX_TOTAL_PIXELS:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Padded image size exceeds server limits")


def require_str(
    body: dict, name: str, *, required: bool, default: str = "", max_len: int = MAX_PROMPT_LEN
) -> str:
    if name not in body or body[name] is None:
        if required:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' is required")
        return default
    value = body[name]
    if not isinstance(value, str):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' must be a string")
    if len(value) > max_len:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' is too long (max {max_len} characters)")
    return value


def optional_int(body: dict, name: str, *, minimum: int, maximum: int) -> int | None:
    if name not in body or body[name] is None:
        return None
    value = body[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' must be an integer")
    if value < minimum or value > maximum:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' must be between {minimum} and {maximum}")
    return value


def optional_float(
    body: dict, name: str, *, minimum: float, maximum: float, default: float
) -> float:
    if name not in body or body[name] is None:
        return default
    value = body[name]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' must be a finite number")
    if value < minimum or value > maximum:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' must be between {minimum} and {maximum}")
    return value


# --------------------------------------------------------------------------
# ComfyUI HTTP client (plain stdlib, no Qt)
# --------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects from anything we talk to, including our own managed ComfyUI."""

    def redirect_request(self, *args, **kwargs):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


class ComfyError(Exception):
    """Communication failure or malformed/unexpected response from ComfyUI -> HTTP 502."""


class ComfyBusyError(Exception):
    """Not enough time budget left, or ComfyUI is not reachable/ready yet -> HTTP 503."""


class ComfyClient:
    """Minimal synchronous client for ComfyUI's HTTP API (no websocket use; we poll /history)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def remaining(deadline: float) -> float:
        return deadline - time.monotonic()

    def _budget(self, deadline: float, default: float) -> float:
        """Timeout to use for one HTTP call: bounded by both a sane per-call
        default and by whatever is left of the request's overall deadline."""
        remaining = self.remaining(deadline)
        if remaining <= 0:
            raise ComfyBusyError("No time budget left for this request")
        return max(0.1, min(default, remaining))

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | bytearray | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = COMFY_DEFAULT_OP_TIMEOUT,
        max_bytes: int = COMFY_JSON_MAX_BYTES,
    ) -> bytes:
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method, headers=headers or {}
        )
        try:
            with _opener.open(req, timeout=timeout) as resp:
                return _read_capped(resp, max_bytes, f"ComfyUI response for {path}")
        except urllib.error.HTTPError as e:
            body = e.read(4096)
            body_text = body.decode("utf-8", "replace")
            raise ComfyError(f"ComfyUI returned HTTP {e.code} for {path}: {body_text[:500]}") from e
        except urllib.error.URLError as e:
            raise ComfyError(f"Could not reach ComfyUI at {self.base_url}{path}: {e.reason}") from e
        except TimeoutError as e:
            raise ComfyError(f"Timed out talking to ComfyUI at {self.base_url}{path}") from e
        except (OSError, ValueError) as e:
            raise ComfyError(f"Error talking to ComfyUI at {self.base_url}{path}: {e}") from e

    def is_reachable(self, timeout: float = 2.0) -> bool:
        try:
            self._request("GET", "/system_stats", timeout=timeout)
            return True
        except ComfyError:
            return False

    def system_stats(self, timeout: float = 5.0) -> dict:
        try:
            return strict_json_loads(
                self._request("GET", "/system_stats", timeout=timeout).decode("utf-8")
            )
        except (ComfyError, ValueError):
            return {}

    def object_info(self, node_class: str, deadline: float) -> dict:
        raw = self._request(
            "GET", f"/object_info/{node_class}", timeout=self._budget(deadline, 10.0)
        )
        try:
            return strict_json_loads(raw.decode("utf-8"))
        except ValueError as e:
            raise ComfyError(f"ComfyUI returned malformed JSON for object_info: {e}") from e

    def select_checkpoint(self, requested: str | None, deadline: float) -> str | None:
        try:
            info = self.object_info("CheckpointLoaderSimple", deadline)
            names = sorted(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0])
        except (ComfyError, ComfyBusyError, KeyError, IndexError, TypeError):
            return None
        if not names:
            return None
        if requested is None:
            return names[0]
        return requested if requested in names else None

    def upload_image(self, filename: str, png_bytes: bytes, deadline: float) -> str:
        boundary = uuid4().hex
        body = bytearray()

        def add_field(name: str, value: str) -> None:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.extend(value.encode())
            body.extend(b"\r\n")

        add_field("overwrite", "true")
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode()
        )
        body.extend(b"Content-Type: image/png\r\n\r\n")
        body.extend(png_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode())

        # `body` (a bytearray) is passed straight through to urllib without an
        # extra bytes() copy -- urllib.request/http.client accept bytes-like
        # objects for `data` directly.
        raw = self._request(
            "POST",
            "/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=self._budget(deadline, 30.0),
        )
        try:
            info = strict_json_loads(raw.decode("utf-8"))
            subfolder = info.get("subfolder") or ""
            name = info["name"]
        except (ValueError, KeyError, AttributeError) as e:
            raise ComfyError(f"ComfyUI returned malformed upload response: {e}") from e
        return f"{subfolder}/{name}" if subfolder else name

    def queue_prompt(self, graph: dict, deadline: float) -> str:
        payload = json.dumps({"prompt": graph, "client_id": f"darask-{uuid4().hex}"}).encode()
        raw = self._request(
            "POST",
            "/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=self._budget(deadline, 15.0),
        )
        try:
            info = strict_json_loads(raw.decode("utf-8"))
        except ValueError as e:
            raise ComfyError(f"ComfyUI returned malformed /prompt response: {e}") from e
        node_errors = info.get("node_errors") or {}
        if node_errors:
            raise ComfyError(f"ComfyUI rejected the workflow: {json.dumps(node_errors)[:1000]}")
        prompt_id = info.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"ComfyUI did not return a prompt_id: {info}")
        return prompt_id

    def wait_for_result(
        self, prompt_id: str, output_node: str, expected_w: int, expected_h: int, deadline: float
    ) -> tuple[bytes, str, str, str]:
        """Poll /history until the job completes, then fetch and validate the
        output image. Returns (png_bytes, filename, subfolder, type_)."""
        while True:
            timeout = self._budget(deadline, 10.0)
            raw = self._request("GET", f"/history/{prompt_id}", timeout=timeout)
            try:
                history = strict_json_loads(raw.decode("utf-8"))
            except ValueError as e:
                raise ComfyError(f"ComfyUI returned malformed /history response: {e}") from e
            entry = history.get(prompt_id) if isinstance(history, dict) else None
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise ComfyError(f"ComfyUI job failed: {messages[:5]}")
                outputs = entry.get("outputs") or {}
                node_out = outputs.get(output_node)
                if node_out and node_out.get("images"):
                    img = node_out["images"][0]
                    try:
                        filename = img["filename"]
                    except (KeyError, TypeError) as e:
                        raise ComfyError(
                            f"ComfyUI history entry missing image filename: {e}"
                        ) from e
                    subfolder = img.get("subfolder", "")
                    type_ = img.get("type", "output")
                    png_bytes = self.fetch_image(
                        filename, subfolder, type_, expected_w, expected_h, deadline
                    )
                    return png_bytes, filename, subfolder, type_
            time.sleep(min(COMFY_POLL_INTERVAL, max(0.0, self.remaining(deadline))))

    def fetch_image(
        self,
        filename: str,
        subfolder: str,
        type_: str,
        expected_w: int,
        expected_h: int,
        deadline: float,
    ) -> bytes:
        query = urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
        data = self._request(
            "GET",
            f"/view?{query}",
            timeout=self._budget(deadline, 30.0),
            max_bytes=COMFY_IMAGE_MAX_BYTES,
        )
        try:
            w, h = png_dimensions(data)
        except PngError as e:
            raise ComfyError(f"ComfyUI returned an invalid image: {e}") from e
        if (w, h) != (expected_w, expected_h):
            raise ComfyError(
                f"ComfyUI returned an image of unexpected size {w}x{h} (expected {expected_w}x{expected_h})"
            )
        return data

    def cancel(self, prompt_id: str) -> None:
        """Best-effort cancel on timeout.

        First try to remove the prompt from ComfyUI's queue (only works if it
        has not started running yet). Then fall back to the global
        /interrupt endpoint, which stops whatever prompt is *currently*
        executing -- not necessarily this one by id, because the ComfyUI
        version this plugin pins does not expose a per-prompt interrupt API.
        This is safe under this plugin's design (darask_server.py serializes
        all requests to a single, dedicated ComfyUI instance -- see
        README_darask.md), but would be wrong against a ComfyUI instance
        shared with other clients.
        """
        try:
            payload = json.dumps({"delete": [prompt_id]}).encode()
            self._request(
                "POST",
                "/queue",
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=5.0,
            )
        except ComfyError:
            pass
        try:
            self._request("POST", "/interrupt", data=b"", timeout=5.0)
        except ComfyError:
            pass


def _read_capped(resp, max_bytes: int, what: str) -> bytes:
    chunks = bytearray()
    while True:
        chunk = resp.read(min(1024 * 1024, max_bytes - len(chunks) + 1))
        if not chunk:
            break
        chunks.extend(chunk)
        if len(chunks) > max_bytes:
            raise ComfyError(f"{what} exceeded the {max_bytes} byte limit")
    return bytes(chunks)


# --------------------------------------------------------------------------
# Workflow graph builders (plain ComfyUI "API format" JSON, standard nodes only)
# --------------------------------------------------------------------------


def build_txt2img_graph(
    checkpoint: str, prompt: str, negative: str, width: int, height: int, seed: int
) -> tuple[dict, str]:
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": DEFAULT_STEPS,
                "cfg": DEFAULT_CFG,
                "sampler_name": DEFAULT_SAMPLER,
                "scheduler": DEFAULT_SCHEDULER,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "darask_generate"},
        },
    }
    return graph, "7"


def build_inpaint_graph(
    checkpoint: str,
    image_ref: str,
    mask_ref: str,
    prompt: str,
    negative: str,
    strength: float,
    seed: int,
) -> tuple[dict, str]:
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_ref}},
        "5": {"class_type": "LoadImageMask", "inputs": {"image": mask_ref, "channel": "red"}},
        "6": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {
                "pixels": ["4", 0],
                "vae": ["1", 2],
                "mask": ["5", 0],
                "grow_mask_by": DEFAULT_INPAINT_GROW_MASK,
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": DEFAULT_STEPS,
                "cfg": DEFAULT_CFG,
                "sampler_name": DEFAULT_SAMPLER,
                "scheduler": DEFAULT_SCHEDULER,
                "denoise": strength,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["6", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "darask_inpaint"},
        },
    }
    return graph, "9"


# --------------------------------------------------------------------------
# ComfyUI process supervisor (optional: darask-plugin.bat asks us to own it)
# --------------------------------------------------------------------------


class ComfyManager:
    """Launches and supervises a ComfyUI child process, if configured to do so."""

    def __init__(
        self,
        comfy_python: Path | None,
        comfy_main: Path | None,
        comfy_port: int,
        log_file: Path | None,
        extra_args: Sequence[str] = (),
    ):
        self.comfy_python = comfy_python
        self.comfy_main = comfy_main
        self.comfy_port = comfy_port
        self.log_file = log_file
        self.extra_args = list(extra_args)
        self.process: subprocess.Popen | None = None
        self.start_time = time.monotonic()
        self._log_fp = None

    @property
    def owns_process(self) -> bool:
        return self.comfy_python is not None and self.comfy_main is not None

    @property
    def comfy_dir(self) -> Path | None:
        return self.comfy_main.parent if self.comfy_main is not None else None

    def start(self) -> None:
        if not self.owns_process:
            return
        assert self.comfy_python is not None and self.comfy_main is not None
        if not self.comfy_python.exists():
            log.error("ComfyUI python not found at %s; will not manage ComfyUI", self.comfy_python)
            return
        if not self.comfy_main.exists():
            log.error("ComfyUI main.py not found at %s; will not manage ComfyUI", self.comfy_main)
            return

        args = self.command_line()
        log.info("Starting ComfyUI: %s", " ".join(args))
        stdout = subprocess.DEVNULL
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            # Kept open for the lifetime of the ComfyUI child process (used as its
            # stdout target below), not a short-lived read; closed in stop().
            self._log_fp = open(self.log_file, "ab")
            stdout = self._log_fp
        self.process = subprocess.Popen(
            args,
            cwd=str(self.comfy_main.parent),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        self.start_time = time.monotonic()

    def command_line(self) -> list[str]:
        assert self.comfy_python is not None and self.comfy_main is not None
        return [
            str(self.comfy_python),
            "-su",
            str(self.comfy_main),
            "--port",
            str(self.comfy_port),
            *self.extra_args,
        ]

    def is_alive(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            log.info("Stopping ComfyUI (pid=%s)", self.process.pid)
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except Exception as e:
                log.warning("ComfyUI did not terminate cleanly (%s); killing it", e)
                try:
                    self.process.kill()
                except Exception:
                    log.exception("Failed to kill ComfyUI process")
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:
                log.exception("Failed to close ComfyUI log file")

    def cleanup_file(self, subdir: str, name_ref: str) -> None:
        """Delete one file we created (upload or SaveImage output). Only ever
        deletes files under ComfyUI's own input/output directory whose name
        starts with 'darask_', so this can never touch anything else."""
        if not self.owns_process or self.comfy_dir is None:
            return
        try:
            base = (self.comfy_dir / subdir).resolve()
            rel = Path(name_ref)
            if not rel.name.startswith("darask_"):
                return
            target = (base / rel).resolve()
            if target.is_relative_to(base) and target.is_file():
                target.unlink()
        except Exception:
            log.exception("Failed to clean up %s/%s", subdir, name_ref)

    def cleanup_stale_files(self, max_age_seconds: float = CLEANUP_MAX_AGE_SECONDS) -> None:
        """Startup sweep: remove leftover darask_* files from a previous run
        that did not shut down cleanly (crash, force-kill)."""
        if not self.owns_process or self.comfy_dir is None:
            return
        now = time.time()
        for subdir in ("input", "output", "temp"):
            d = self.comfy_dir / subdir
            if not d.is_dir():
                continue
            for f in d.glob("darask_*"):
                try:
                    if f.is_file() and (now - f.stat().st_mtime) > max_age_seconds:
                        f.unlink()
                        log.info("Removed stale leftover file %s", f)
                except Exception:
                    log.exception("Failed to remove stale file %s", f)


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------


@dataclass
class ServerContext:
    comfy: ComfyClient
    comfy_manager: ComfyManager
    generation_timeout: float
    lock: threading.Lock
    server_start_time: float
    requested_checkpoint: str | None = None
    checkpoint_error: str | None = field(default=None)


class Handler(BaseHTTPRequestHandler):
    server_version = f"{DARASK_PLUGIN_NAME}/{DARASK_PLUGIN_VERSION}"
    protocol_version = "HTTP/1.1"
    timeout = SOCKET_READ_TIMEOUT  # Slowloris defense: socketserver applies this to the raw socket

    # Silence default per-request stderr logging; we log through `logging` instead.
    def log_message(self, format: str, *args) -> None:
        log.debug("%s - %s", self.address_string(), format % args)

    @property
    def ctx(self) -> ServerContext:
        return self.server.ctx  # type: ignore[attr-defined]

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_png(self, png_bytes: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png_bytes)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(png_bytes)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _check_common_security(self) -> None:
        """Defense against a malicious/compromised web page abusing the
        browser as a confused deputy against this localhost server (DNS
        rebinding, plain CSRF-style fetch(), etc.)."""
        host_header = self.headers.get("Host")
        if host_header is None:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Host header is required")
        port = self.server.server_port  # type: ignore[attr-defined]
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if host_header.strip().lower() not in allowed_hosts:
            raise ApiError(HTTPStatus.FORBIDDEN, "Unexpected Host header")
        if self.headers.get("Origin") is not None:
            raise ApiError(HTTPStatus.FORBIDDEN, "Cross-origin requests are not allowed")
        sec_fetch_site = self.headers.get("Sec-Fetch-Site")
        if sec_fetch_site is not None and sec_fetch_site.strip().lower() == "cross-site":
            raise ApiError(HTTPStatus.FORBIDDEN, "Cross-site requests are not allowed")
        if self.headers.get("Transfer-Encoding") is not None:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Chunked transfer encoding is not supported")
        if len(self.headers.get_all("Content-Length") or []) > 1:
            raise ApiError(
                HTTPStatus.BAD_REQUEST, "Duplicate Content-Length headers are not allowed"
            )

    def _read_json_body(self) -> dict:
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";")[0].strip().lower() != "application/json":
            raise ApiError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json"
            )

        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise ApiError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
        try:
            length = int(length_header)
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length") from None
        if length < 0 or length > MAX_REQUEST_BODY_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body too large")

        remaining = length
        chunks = []
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST, "Connection closed before body was fully sent"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        try:
            body = strict_json_loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Request body is not valid JSON") from None
        if not isinstance(body, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object")
        return body

    # -- routing -----------------------------------------------------------

    def do_OPTIONS(self) -> None:
        self.close_connection = True
        # No CORS support at all: any preflight is rejected outright.
        self._send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed")

    def do_GET(self) -> None:
        self.close_connection = True
        try:
            self._check_common_security()
            if self.path == "/api/v1/health":
                self._handle_health()
            else:
                raise ApiError(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except ApiError as e:
            self._send_error_json(e.status, e.message)
        except Exception:
            log.exception("Unhandled error in GET %s", self.path)
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error")

    def do_POST(self) -> None:
        self.close_connection = True
        try:
            self._check_common_security()
            if self.path == "/api/v1/generate":
                self._handle_generate()
            elif self.path == "/api/v1/inpaint":
                self._handle_inpaint()
            else:
                raise ApiError(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        except ApiError as e:
            self._send_error_json(e.status, e.message)
        except ComfyBusyError as e:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(e))
        except ComfyError as e:
            self._send_error_json(HTTPStatus.BAD_GATEWAY, str(e))
        except Exception:
            log.exception("Unhandled error in POST %s", self.path)
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error")

    # -- endpoints -----------------------------------------------------------

    def _handle_health(self) -> None:
        ctx = self.ctx
        stats = ctx.comfy.system_stats(timeout=2.0)
        reachable = bool(stats)
        detail = None

        if reachable:
            backend = "ready"
        elif ctx.comfy_manager.owns_process and not ctx.comfy_manager.is_alive():
            backend = "error"
            detail = "ComfyUI process exited unexpectedly"
        else:
            elapsed = time.monotonic() - ctx.server_start_time
            if elapsed < COMFY_STARTUP_GRACE:
                backend = "starting"
            else:
                backend = "error"
                detail = f"ComfyUI not reachable after {int(elapsed)}s"

        comfy_version = None
        try:
            comfy_version = stats.get("system", {}).get("comfyui_version")
        except AttributeError:
            pass
        engine = f"{DARASK_PLUGIN_VERSION} (ComfyUI {comfy_version or '?'})"

        model = None
        if reachable:
            deadline = time.monotonic() + 5.0
            model = ctx.comfy.select_checkpoint(ctx.requested_checkpoint, deadline)
            if model is None and ctx.requested_checkpoint is not None:
                backend = "error"
                detail = (
                    f"Requested checkpoint '{ctx.requested_checkpoint}' is not installed in ComfyUI"
                )

        payload = {
            "plugin": DARASK_PLUGIN_NAME,
            "api": API_VERSION,
            "engine": engine,
            "backend": backend,
            "model": model,
        }
        if detail is not None:
            # Additive field; darask-paint's hand-written JSON parser is
            # expected to only read known keys and ignore the rest (spec 55.1).
            payload["detail"] = detail
        self._send_json(HTTPStatus.OK, payload)

    def _wait_for_backend(self, ctx: ServerContext, deadline: float) -> None:
        while True:
            if ctx.comfy_manager.owns_process and not ctx.comfy_manager.is_alive():
                raise ComfyBusyError("ComfyUI process is not running")
            if ctx.comfy.is_reachable(timeout=2.0):
                return
            if time.monotonic() >= deadline:
                raise ComfyBusyError("ComfyUI backend is not ready")
            time.sleep(1.0)

    def _acquire_or_busy(self, ctx: ServerContext) -> None:
        if not ctx.lock.acquire(timeout=LOCK_WAIT_TIMEOUT):
            raise ComfyBusyError("Server is busy with another job")

    def _resolve_checkpoint(self, ctx: ServerContext, deadline: float) -> str:
        checkpoint = ctx.comfy.select_checkpoint(ctx.requested_checkpoint, deadline)
        if checkpoint is None:
            if ctx.requested_checkpoint is not None:
                raise ComfyBusyError(
                    f"Requested checkpoint '{ctx.requested_checkpoint}' is not installed"
                )
            raise ComfyBusyError("No checkpoint model is installed in ComfyUI")
        return checkpoint

    def _handle_generate(self) -> None:
        body = self._read_json_body()
        prompt = require_str(body, "prompt", required=True)
        if not prompt.strip():
            raise ApiError(HTTPStatus.BAD_REQUEST, "'prompt' must not be empty")
        negative = require_str(body, "negative", required=False, default="")
        width = body.get("width")
        height = body.get("height")
        validate_extent(width, height)
        seed = optional_int(body, "seed", minimum=0, maximum=2**32 - 1)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        padded_w, padded_h = next_multiple_of_8(width), next_multiple_of_8(height)
        validate_padded_extent(padded_w, padded_h)

        ctx = self.ctx
        deadline = time.monotonic() + ctx.generation_timeout
        self._acquire_or_busy(ctx)
        prompt_id: str | None = None
        output_ref: tuple[str, str] | None = None  # (subdir, name_ref)
        try:
            self._wait_for_backend(ctx, deadline)
            checkpoint = self._resolve_checkpoint(ctx, deadline)
            graph, output_node = build_txt2img_graph(
                checkpoint, prompt, negative, padded_w, padded_h, seed
            )
            prompt_id = ctx.comfy.queue_prompt(graph, deadline)
            png_bytes, out_name, out_subfolder, out_type = ctx.comfy.wait_for_result(
                prompt_id, output_node, padded_w, padded_h, deadline
            )
            output_ref = (out_type, f"{out_subfolder}/{out_name}" if out_subfolder else out_name)
            if (padded_w, padded_h) != (width, height):
                png_bytes = png_crop_to(png_bytes, width, height)
            self._send_png(png_bytes)
        except (ComfyError, ComfyBusyError):
            if prompt_id is not None:
                ctx.comfy.cancel(prompt_id)
            raise
        finally:
            if output_ref is not None:
                ctx.comfy_manager.cleanup_file(*output_ref)
            ctx.lock.release()

    def _handle_inpaint(self) -> None:
        body = self._read_json_body()
        image_bytes = decode_base64_image("image", body.get("image"))
        mask_bytes = decode_base64_image("mask", body.get("mask"))
        prompt = require_str(body, "prompt", required=True)
        if not prompt.strip():
            raise ApiError(HTTPStatus.BAD_REQUEST, "'prompt' must not be empty")
        negative = require_str(body, "negative", required=False, default="")
        strength = optional_float(body, "strength", minimum=0.01, maximum=1.0, default=1.0)

        try:
            image_w, image_h = png_dimensions(image_bytes)
            mask_w, mask_h = png_dimensions(mask_bytes)
        except PngError as e:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(e)) from None
        validate_extent(image_w, image_h)
        if (image_w, image_h) != (mask_w, mask_h):
            raise ApiError(
                HTTPStatus.BAD_REQUEST, "'image' and 'mask' must have the same dimensions"
            )

        padded_w, padded_h = next_multiple_of_8(image_w), next_multiple_of_8(image_h)
        validate_padded_extent(padded_w, padded_h)
        try:
            if (padded_w, padded_h) != (image_w, image_h):
                image_bytes = png_pad_to(image_bytes, padded_w, padded_h)
                mask_bytes = png_pad_to(mask_bytes, padded_w, padded_h)
        except PngError as e:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Failed to process image/mask: {e}") from None

        seed = optional_int(body, "seed", minimum=0, maximum=2**32 - 1)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        ctx = self.ctx
        deadline = time.monotonic() + ctx.generation_timeout
        self._acquire_or_busy(ctx)
        prompt_id: str | None = None
        cleanup_refs: list[tuple[str, str]] = []
        try:
            self._wait_for_backend(ctx, deadline)
            checkpoint = self._resolve_checkpoint(ctx, deadline)
            image_ref = ctx.comfy.upload_image(f"darask_{uuid4().hex}.png", image_bytes, deadline)
            cleanup_refs.append(("input", image_ref))
            mask_ref = ctx.comfy.upload_image(
                f"darask_{uuid4().hex}_mask.png", mask_bytes, deadline
            )
            cleanup_refs.append(("input", mask_ref))
            graph, output_node = build_inpaint_graph(
                checkpoint, image_ref, mask_ref, prompt, negative, strength, seed
            )
            prompt_id = ctx.comfy.queue_prompt(graph, deadline)
            png_bytes, out_name, out_subfolder, out_type = ctx.comfy.wait_for_result(
                prompt_id, output_node, padded_w, padded_h, deadline
            )
            cleanup_refs.append((
                out_type,
                f"{out_subfolder}/{out_name}" if out_subfolder else out_name,
            ))
            if (padded_w, padded_h) != (image_w, image_h):
                png_bytes = png_crop_to(png_bytes, image_w, image_h)
            self._send_png(png_bytes)
        except (ComfyError, ComfyBusyError):
            if prompt_id is not None:
                ctx.comfy.cancel(prompt_id)
            raise
        finally:
            for subdir, ref in cleanup_refs:
                ctx.comfy_manager.cleanup_file(subdir, ref)
            ctx.lock.release()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        ctx: ServerContext,
        max_concurrent_handlers: int = DEFAULT_MAX_CONCURRENT_HANDLERS,
    ):
        super().__init__(address, handler)
        self.ctx = ctx
        self._handler_semaphore = threading.Semaphore(max_concurrent_handlers)

    # Overrides ThreadingMixIn.process_request/process_request_thread to add
    # an admission-control semaphore, protecting against thread-exhaustion
    # DoS from a client opening many slow/idle connections (see also
    # Handler.timeout, which caps how long any single connection may stall).
    def process_request(self, request, client_address) -> None:
        if not self._handler_semaphore.acquire(timeout=1.0):
            log.warning("Too many concurrent connections; rejecting %s", client_address)
            try:
                self.shutdown_request(request)
            except Exception:
                log.exception("Failed to close rejected connection from %s", client_address)
            return
        t = threading.Thread(target=self._process_request_thread, args=(request, client_address))
        t.daemon = self.daemon_threads
        t.start()

    def _process_request_thread(self, request, client_address) -> None:
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)
            self._handler_semaphore.release()

    def handle_error(self, request, client_address) -> None:
        log.exception("Error handling request from %s", client_address)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def validate_comfy_url(url: str) -> str:
    """Restrict --comfy-url to plain http://127.0.0.1:<port> (no userinfo,
    no fragment, no other host). This is a startup-time check; a violation
    is a configuration error, not a runtime API error."""
    parts = urlsplit(url)
    if parts.scheme != "http":
        raise ValueError(f"--comfy-url must use http:// (got scheme={parts.scheme!r})")
    if parts.username is not None or parts.password is not None:
        raise ValueError("--comfy-url must not contain userinfo (user:pass@)")
    if parts.hostname != "127.0.0.1":
        raise ValueError(f"--comfy-url host must be 127.0.0.1 (got {parts.hostname!r})")
    if parts.fragment:
        raise ValueError("--comfy-url must not contain a fragment")
    port = parts.port
    if port is None or not (1 <= port <= 65535):
        raise ValueError("--comfy-url must specify a port between 1 and 65535")
    if parts.path not in ("", "/") or parts.query:
        raise ValueError("--comfy-url must not contain a path or query string")
    return f"http://127.0.0.1:{port}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Darask Paint AI Diffusion plugin server")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to listen on (127.0.0.1 only)"
    )
    parser.add_argument(
        "--comfy-url",
        default="http://127.0.0.1:8188",
        help="Base URL of the ComfyUI HTTP API to use",
    )
    parser.add_argument(
        "--comfy-port", type=int, default=8188, help="Port used when launching ComfyUI ourselves"
    )
    parser.add_argument(
        "--comfy-python",
        type=Path,
        default=None,
        help="Path to the ComfyUI venv's python.exe (optional)",
    )
    parser.add_argument(
        "--comfy-main", type=Path, default=None, help="Path to ComfyUI's main.py (optional)"
    )
    parser.add_argument(
        "--comfy-log",
        type=Path,
        default=None,
        help="File to redirect the managed ComfyUI's output to",
    )
    parser.add_argument(
        "--comfy-arg",
        dest="comfy_args",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument forwarded to the managed ComfyUI (repeatable), e.g. --comfy-arg=--cpu",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Exact checkpoint filename to require. If omitted, the first checkpoint in "
        "ComfyUI's sorted list of installed checkpoints is used.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_GENERATION_TIMEOUT,
        help="Per-request generation timeout in seconds",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_HANDLERS,
        help="Maximum number of simultaneous HTTP connections",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite, positive number")
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    try:
        args.comfy_url = validate_comfy_url(args.comfy_url)
    except ValueError as e:
        parser.error(str(e))
    return args


def main(argv: list[str] | None = None) -> NoReturn:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    comfy_url = args.comfy_url
    if args.comfy_python and args.comfy_main:
        comfy_url = validate_comfy_url(f"http://127.0.0.1:{args.comfy_port}")

    comfy_manager = ComfyManager(
        args.comfy_python, args.comfy_main, args.comfy_port, args.comfy_log, args.comfy_args
    )
    comfy_client = ComfyClient(comfy_url)

    ctx = ServerContext(
        comfy=comfy_client,
        comfy_manager=comfy_manager,
        generation_timeout=args.timeout,
        lock=threading.Lock(),
        server_start_time=time.monotonic(),
        requested_checkpoint=args.checkpoint,
    )

    def shutdown(*_args) -> None:
        log.info("Shutting down")
        comfy_manager.stop()

    def raise_keyboard_interrupt(_signum, _frame) -> None:
        raise KeyboardInterrupt()

    atexit.register(shutdown)
    # SIGINT (Ctrl+C) already raises KeyboardInterrupt via Python's default handler.
    # SIGBREAK (Windows Ctrl+Break) does not, so wire it up explicitly for the same
    # clean-shutdown path used by serve_forever()'s KeyboardInterrupt handling below.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, raise_keyboard_interrupt)

    comfy_manager.cleanup_stale_files()
    if comfy_manager.owns_process:
        comfy_manager.start()

    httpd = Server(
        (DEFAULT_HOST, args.port), Handler, ctx, max_concurrent_handlers=args.max_concurrent
    )
    log.info("darask-ai-diffusion server listening on http://%s:%d", DEFAULT_HOST, args.port)
    log.info("Talking to ComfyUI at %s", comfy_url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
