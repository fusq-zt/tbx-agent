from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError


class ImageValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    image: Image.Image
    sha256: str
    width: int
    height: int
    source_format: str
    input_transform_id: str
    quality_status: str
    quality_warnings: tuple[str, ...]
    artifact_bytes: bytes

    def persist_png(self, artifact_root: Path, case_id: str) -> Path:
        case_root = artifact_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        target = case_root / f"cxr-{self.sha256[:16]}.png"
        self.image.save(target, format="PNG", optimize=True)
        return target

    def persist_original(self, payload: bytes, artifact_root: Path, case_id: str) -> Path:
        """Persist only the normalized metadata-free PNG review artifact."""

        del payload
        case_root = artifact_root / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        target = case_root / f"upload-{self.sha256[:16]}.png"
        if target.exists():
            existing = hashlib.sha256(target.read_bytes()).hexdigest()
            if existing != self.sha256:
                raise ImageValidationError("同名图像制品的完整性校验失败。")
        else:
            target.write_bytes(self.artifact_bytes)
        return target


def _canonical_png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    # Fixed parameters and no metadata make the DICOM-derived artifact stable and
    # prevent patient/study tags from crossing the validation boundary.
    image.convert("RGB").save(output, format="PNG", optimize=False, compress_level=6)
    return output.getvalue()


def _single_dicom_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    try:
        # pydicom MultiValue is sequence-like but is intentionally an optional import.
        if value.__class__.__name__ == "MultiValue":
            return value[0] if len(value) else None
    except (AttributeError, TypeError):
        return value
    return value


def _decode_dicom(payload: bytes) -> tuple[Image.Image, tuple[str, ...]]:
    """Decode a conservative single-frame chest-radiograph DICOM contract.

    The decoder does not claim diagnostic image quality or de-identification. It
    accepts only uncompressed, single-frame, monochrome CR/DX objects and returns
    pixels without copying DICOM tags into the derived raster.
    """

    try:
        import numpy as np
        import pydicom
    except ImportError as exc:
        raise ImageValidationError(
            "检测到 DICOM；请在 API 环境安装项目的 dicom 可选依赖。"
        ) from exc

    try:
        dataset = pydicom.dcmread(io.BytesIO(payload), force=False)
    except Exception as exc:
        raise ImageValidationError("无法安全解析该 DICOM 文件。") from exc

    modality = str(getattr(dataset, "Modality", "")).strip().upper()
    if modality not in {"CR", "DX"}:
        raise ImageValidationError("DICOM 仅接受单帧胸部 CR/DX 放射影像。")
    frames_raw = _single_dicom_value(getattr(dataset, "NumberOfFrames", 1))
    try:
        frames = int(frames_raw or 1)
    except (TypeError, ValueError) as exc:
        raise ImageValidationError("DICOM NumberOfFrames 无效。") from exc
    if frames != 1:
        raise ImageValidationError("DICOM 仅接受单帧影像。")

    transfer_syntax = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", None)
    if transfer_syntax is None:
        raise ImageValidationError("DICOM 缺少 TransferSyntaxUID，无法验证像素编码。")
    if bool(getattr(transfer_syntax, "is_compressed", False)):
        raise ImageValidationError("暂不接受压缩 DICOM；请先在受控流程中转为无损单帧 CR/DX。")

    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).strip().upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ImageValidationError("DICOM 仅接受 MONOCHROME1/2 灰度像素。")
    try:
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)
        samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ImageValidationError("DICOM 像素尺寸字段缺失或无效。") from exc
    if rows <= 0 or columns <= 0 or rows * columns > 100_000_000:
        raise ImageValidationError("DICOM 解码后图像尺寸无效或过大。")
    if samples_per_pixel != 1:
        raise ImageValidationError("DICOM 仅接受单通道灰度影像。")
    if "PixelData" not in dataset:
        raise ImageValidationError("DICOM 不含可用 PixelData。")

    try:
        pixels = np.asarray(dataset.pixel_array)
    except Exception as exc:
        raise ImageValidationError("DICOM 像素数据无法使用已安装的安全解码器读取。") from exc
    if pixels.ndim != 2 or pixels.shape != (rows, columns):
        raise ImageValidationError("DICOM 解码像素形状与 Rows/Columns 不一致。")
    values = pixels.astype(np.float64, copy=False)
    slope = float(_single_dicom_value(getattr(dataset, "RescaleSlope", 1.0)) or 1.0)
    intercept = float(_single_dicom_value(getattr(dataset, "RescaleIntercept", 0.0)) or 0.0)
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope == 0:
        raise ImageValidationError("DICOM 灰度缩放参数无效。")
    values = values * slope + intercept
    if not np.isfinite(values).all():
        raise ImageValidationError("DICOM 像素包含非有限值。")

    lower: float
    upper: float
    center_raw = _single_dicom_value(getattr(dataset, "WindowCenter", None))
    width_raw = _single_dicom_value(getattr(dataset, "WindowWidth", None))
    try:
        center = float(center_raw) if center_raw is not None else None
        width = float(width_raw) if width_raw is not None else None
    except (TypeError, ValueError):
        center = width = None
    if center is not None and width is not None and np.isfinite(center) and width > 1:
        lower, upper = center - width / 2.0, center + width / 2.0
    else:
        lower, upper = (float(item) for item in np.percentile(values, (0.5, 99.5)))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ImageValidationError("DICOM 像素动态范围不足，无法形成可用胸片。")
    normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    if photometric == "MONOCHROME1":
        normalized = 1.0 - normalized
    raster = Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="L").convert(
        "RGB"
    )

    warnings: list[str] = []
    body_part = str(getattr(dataset, "BodyPartExamined", "")).strip().upper()
    if body_part not in {"CHEST", "THORAX", "LUNG"}:
        warnings.append("dicom_chest_body_part_not_verified")
    view = str(getattr(dataset, "ViewPosition", "")).strip().upper()
    if view not in {"AP", "PA"}:
        warnings.append("dicom_frontal_view_not_verified")
    burned_in = str(getattr(dataset, "BurnedInAnnotation", "")).strip().upper()
    if burned_in == "YES":
        raise ImageValidationError("DICOM 声明含烧录标注；请先完成去标识化并重新导出。")
    if burned_in != "NO":
        warnings.append("dicom_burned_in_annotation_not_excluded")
    return raster, tuple(warnings)


def validate_image(payload: bytes, *, max_bytes: int) -> ValidatedImage:
    """Decode a PNG/JPEG or conservative DICOM input and collect warnings.

    This validates transport and coarse image suitability only.  It does not
    establish that the image is a diagnostic-quality frontal chest radiograph.
    """

    if not payload:
        raise ImageValidationError("上传文件为空。")
    if len(payload) > max_bytes:
        raise ImageValidationError(f"文件超过 {max_bytes} 字节上传限制。")

    dicom_part10 = len(payload) >= 132 and payload[128:132] == b"DICM"
    warnings: list[str] = []
    if dicom_part10:
        image, dicom_warnings = _decode_dicom(payload)
        source_format = "DICOM"
        input_transform_id = "dicom-crdx-windowed-rgb-v1"
        artifact_bytes = _canonical_png(image)
        digest = hashlib.sha256(artifact_bytes).hexdigest()
        warnings.extend(dicom_warnings)
        width, height = image.size
    else:
        try:
            with Image.open(io.BytesIO(payload)) as probe:
                source_format = (probe.format or "").upper()
                probe.verify()
            if source_format not in {"JPEG", "PNG"}:
                raise ImageValidationError("仅接受 PNG/JPEG 或带 DICM 前导的单帧 CR/DX。")
            with Image.open(io.BytesIO(payload)) as decoded:
                decoded = ImageOps.exif_transpose(decoded)
                width, height = decoded.size
                if width <= 0 or height <= 0:
                    raise ImageValidationError("图像尺寸无效。")
                if width * height > 100_000_000:
                    raise ImageValidationError("解码后图像过大。")
                image = decoded.convert("RGB").copy()
                input_transform_id = "raster-exif-transpose-rgb-v1"
                artifact_bytes = _canonical_png(image)
                digest = hashlib.sha256(artifact_bytes).hexdigest()
        except ImageValidationError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageValidationError("无法安全解码该图像。") from exc

    if min(width, height) < 512:
        warnings.append("resolution_below_512")
    if (width, height) != (512, 512):
        warnings.append("outside_rank03_validated_512x512_domain")
    ratio = width / height
    if not 0.65 <= ratio <= 1.35:
        warnings.append("unusual_aspect_ratio")
    extrema = image.convert("L").getextrema()
    if extrema is not None and extrema[1] - extrema[0] < 20:
        warnings.append("very_low_dynamic_range")

    return ValidatedImage(
        image=image,
        sha256=digest,
        width=width,
        height=height,
        source_format=source_format,
        input_transform_id=input_transform_id,
        quality_status="warning" if warnings else "transport_valid",
        quality_warnings=tuple(warnings),
        artifact_bytes=artifact_bytes,
    )
