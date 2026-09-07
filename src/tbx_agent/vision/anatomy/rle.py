"""Compact, deterministic binary-mask serialization."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterable, Sequence

from .models import CompactRLE


class MaskEncodingError(ValueError):
    pass


def _shape_and_flatten(mask: object) -> tuple[int, int, list[bool]]:
    """Convert a two-dimensional array-like object without importing NumPy eagerly."""

    shape = getattr(mask, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise MaskEncodingError("a binary mask must be two-dimensional")
        height, width = int(shape[0]), int(shape[1])
        try:
            values = [bool(value) for row in mask for value in row]
        except TypeError as exc:
            raise MaskEncodingError("mask rows must be iterable") from exc
    else:
        if not isinstance(mask, Sequence) or not mask:
            raise MaskEncodingError("a binary mask must contain at least one row")
        rows = list(mask)
        if not all(isinstance(row, Sequence) for row in rows):
            raise MaskEncodingError("mask rows must be sequences")
        height = len(rows)
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise MaskEncodingError("mask rows must have a consistent non-zero width")
        values = [bool(value) for row in rows for value in row]
    if width <= 0 or height <= 0 or len(values) != width * height:
        raise MaskEncodingError("mask shape is invalid")
    return width, height, values


def _uvarint_encode(values: Iterable[int]) -> bytes:
    output = bytearray()
    for raw_value in values:
        value = int(raw_value)
        if value < 0:
            raise MaskEncodingError("run lengths cannot be negative")
        while value >= 0x80:
            output.append((value & 0x7F) | 0x80)
            value >>= 7
        output.append(value)
    return bytes(output)


def _uvarint_decode(payload: bytes) -> list[int]:
    values: list[int] = []
    value = 0
    shift = 0
    for byte in payload:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            if shift > 63:
                raise MaskEncodingError("RLE varint is too large")
        else:
            values.append(value)
            value = 0
            shift = 0
    if shift:
        raise MaskEncodingError("RLE varint is truncated")
    return values


def encode_binary_mask(mask: object) -> CompactRLE:
    width, height, values = _shape_and_flatten(mask)
    runs: list[int] = []
    active = False
    length = 0
    foreground = 0
    for value in values:
        foreground += int(value)
        if value == active:
            length += 1
        else:
            runs.append(length)
            active = value
            length = 1
    runs.append(length)
    binary = _uvarint_encode(runs)
    digest_payload = bytes(int(value) for value in values)
    return CompactRLE(
        width=width,
        height=height,
        counts_b64=base64.urlsafe_b64encode(binary).decode("ascii"),
        foreground_pixels=foreground,
        mask_sha256=hashlib.sha256(digest_payload).hexdigest(),
    )


def decode_binary_mask(payload: CompactRLE) -> list[list[bool]]:
    try:
        binary = base64.b64decode(payload.counts_b64, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise MaskEncodingError("RLE counts are not valid URL-safe base64") from exc
    runs = _uvarint_decode(binary)
    expected = payload.width * payload.height
    if sum(runs) != expected:
        raise MaskEncodingError("RLE run lengths do not match the declared canvas")
    values: list[bool] = []
    active = False
    for length in runs:
        values.extend([active] * length)
        active = not active
    if sum(values) != payload.foreground_pixels:
        raise MaskEncodingError("RLE foreground count does not match metadata")
    digest = hashlib.sha256(bytes(int(value) for value in values)).hexdigest()
    if digest != payload.mask_sha256:
        raise MaskEncodingError("RLE mask digest mismatch")
    return [
        values[offset : offset + payload.width]
        for offset in range(0, expected, payload.width)
    ]

