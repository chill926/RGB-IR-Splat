"""FLIR RJPEG decoding and camera-response coordinates (NumPy/Pillow only).

FFF offsets follow ExifTool's FLIR tag documentation:
https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/FLIR.pm
The blackbody calibration is Q(T)=R1/[R2*(exp(B/T)-F)], Q_observed=DN+O.
Q is a camera-equivalent signal, not SI band radiance or surface-temperature GT.
No emissivity, reflected-temperature, atmosphere, or window correction is applied.
"""
import hashlib
import io
import math
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
from PIL import Image


CALIBRATION_KEYS = ("PlanckR1", "PlanckR2", "PlanckB", "PlanckF", "PlanckO")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_calibration(calibration):
    values = {key: float(calibration[key]) for key in CALIBRATION_KEYS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite FLIR calibration coefficient")
    if min(values[key] for key in ("PlanckR1", "PlanckR2", "PlanckB")) <= 0:
        raise ValueError("FLIR PlanckR1/PlanckR2/PlanckB must be positive")
    return values


def blackbody_signal(temperature_K, calibration):
    c = validate_calibration(calibration)
    temperature = np.asarray(temperature_K, dtype=np.float64)
    if not np.isfinite(temperature).all() or np.any(temperature <= 0):
        raise ValueError("Blackbody temperature must be finite and positive")
    denominator = c["PlanckR2"] * (np.exp(c["PlanckB"] / temperature) - c["PlanckF"])
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("Temperature is outside the valid camera-response range")
    return c["PlanckR1"] / denominator


def signal_to_apparent_temperature(signal, calibration):
    c = validate_calibration(calibration)
    q = np.asarray(signal, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = c["PlanckB"] / np.log(c["PlanckR1"] / (c["PlanckR2"] * q) + c["PlanckF"])
    return np.where((q > 0) & np.isfinite(result) & (result > 0), result, np.nan)


def normalize_signal(signal, calibration, temp_min, temp_max):
    if not 0 < float(temp_min) < float(temp_max):
        raise ValueError("Invalid normalization temperature bounds")
    lo, hi = blackbody_signal([temp_min, temp_max], calibration)
    value = (np.asarray(signal, dtype=np.float64) - lo) / (hi - lo)
    if not np.isfinite(value).all() or value.min() < -1e-6 or value.max() > 1 + 1e-6:
        raise ValueError("Radiometric observation is outside the temperature bounds; widen --temp_min/--temp_max")
    return np.clip(value, 0, 1).astype(np.float32)


def _fff_from_jpeg(data):
    if data[:2] != b"\xff\xd8":
        raise ValueError("Not a JPEG file")
    pieces, pos = {}, 2
    while pos < len(data):
        if data[pos] != 255:
            raise ValueError("Malformed JPEG marker")
        while pos < len(data) and data[pos] == 255:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker in (0xDA, 0xD9):
            break
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            continue
        if pos + 2 > len(data):
            raise ValueError("Truncated JPEG segment")
        size = struct.unpack_from(">H", data, pos)[0]
        if size < 2 or pos + size > len(data):
            raise ValueError("Truncated JPEG segment")
        payload = data[pos + 2:pos + size]
        pos += size
        if marker == 0xE1 and payload.startswith(b"FLIR\x00"):
            if len(payload) < 8:
                raise ValueError("Truncated FLIR APP1 header")
            index = payload[6]
            if index in pieces:
                raise ValueError("Duplicate FLIR APP1 segment")
            pieces[index] = payload[8:]
    if not pieces:
        raise ValueError("JPEG has no FLIR radiometric payload; a pseudo-color export is insufficient")
    order = sorted(pieces)
    if order != list(range(order[0], order[-1] + 1)):
        raise ValueError("Missing FLIR APP1 segment")
    return b"".join(pieces[index] for index in order)


def _records(fff, include_display=False):
    if len(fff) < 64 or fff[:4] not in (b"FFF\x00", b"AFF\x00"):
        raise ValueError("Missing or truncated FLIR FFF header")
    endians = [endian for endian in ("<", ">")
               if 100 <= struct.unpack_from(endian + "I", fff, 0x14)[0] < 200]
    if len(endians) != 1:
        raise ValueError("Unsupported FLIR FFF version")
    endian = endians[0]
    offset, count = struct.unpack_from(endian + "II", fff, 0x18)
    if offset < 64 or count > 10000 or offset + count * 32 > len(fff):
        raise ValueError("Truncated FLIR directory")
    records = {}
    for index in range(count):
        entry = offset + 32 * index
        kind = struct.unpack_from(endian + "H", fff, entry)[0]
        start, length = struct.unpack_from(endian + "II", fff, entry + 12)
        if start + length > len(fff):
            raise ValueError("Truncated FLIR record")
        if kind in ((1, 0x20, 0x22) if include_display else (1, 0x20)):
            if kind in records:
                raise ValueError("Duplicate FLIR raw/calibration record")
            records[kind] = fff[start:start + length]
    if not all(kind in records for kind in (1, 0x20)):
        raise ValueError("FLIR raw signal or camera calibration is missing")
    return records


def read_flir_display_metadata(path):
    """Read palette/window metadata without decoding pixels or fitting to them.

    These settings condition display only. They are not surface-temperature
    truth or an exact description of the dataset's exported tone mapping.
    """
    records = _records(_fff_from_jpeg(Path(path).read_bytes()), include_display=True)
    camera, palette = records[0x20], records.get(0x22)
    if palette is None or len(palette) < 112:
        raise ValueError("FLIR JPEG has no usable display palette")
    candidates = [struct.unpack_from(endian + "I", palette)[0] for endian in ("<", ">")]
    counts = [count for count in candidates if 2 <= count <= 4096 and 112 + count * 3 <= len(palette)]
    if len(counts) != 1:
        raise ValueError("Invalid FLIR palette length")
    endian = _record_endian(camera)
    metadata = _camera_metadata(camera)
    span = int(struct.unpack_from(endian + "H", camera, 0x33C)[0])
    if span <= 0:
        raise ValueError("FLIR RawValueRange is not a positive display-window width")
    colors = np.frombuffer(palette[112:112 + counts[0] * 3], dtype=np.uint8).reshape(-1, 3)
    return {"palette_ycrcb": colors.tolist(),
            "palette_sha256": hashlib.sha256(colors.tobytes()).hexdigest(),
            "palette_name": palette[80:112].split(b"\x00", 1)[0].decode("ascii", errors="replace"),
            "palette_method": int(palette[26]), "palette_stretch": int(palette[27]),
            "raw_value_median": metadata["RawValueMedian"], "raw_value_range": span,
            "signal_window_center": float(metadata["RawValueMedian"] + metadata["PlanckO"])}


def _record_endian(record):
    if record[:2] == b"\x02\x00":
        return "<"
    if record[:2] == b"\x00\x02":
        return ">"
    raise ValueError("Invalid FLIR record byte-order marker")


def _camera_metadata(record):
    if len(record) < 0x340:
        raise ValueError("Truncated FLIR CameraInfo")
    endian = _record_endian(record)
    floats = {"Emissivity": 0x20, "ObjectDistance": 0x24,
              "ReflectedTemperature_K": 0x28, "AtmosphericTemperature_K": 0x2C,
              "IRWindowTemperature_K": 0x30, "IRWindowTransmission": 0x34,
              "RelativeHumidity": 0x3C, "PlanckR1": 0x58, "PlanckB": 0x5C,
              "PlanckF": 0x60, "PlanckR2": 0x30C}
    metadata = {name: float(struct.unpack_from(endian + "f", record, offset)[0])
                for name, offset in floats.items()}
    metadata["PlanckO"] = int(struct.unpack_from(endian + "i", record, 0x308)[0])
    metadata["RawValueMedian"] = int(struct.unpack_from(endian + "H", record, 0x338)[0])
    metadata["CameraModel"] = record[0xD4:0xF4].split(b"\x00", 1)[0].decode("ascii", errors="replace")
    validate_calibration(metadata)
    if not all(math.isfinite(value) for value in metadata.values() if isinstance(value, (float, int))):
        raise ValueError("Non-finite FLIR CameraInfo")
    return metadata


def _byte_order_score(raw, metadata):
    q = raw.astype(np.float64) + metadata["PlanckO"]
    temperature = signal_to_apparent_temperature(q, metadata)
    plausible = np.isfinite(temperature) & (temperature > 150) & (temperature < 2000)
    span = max(float(np.percentile(raw, 95) - np.percentile(raw, 5)), 1.0)
    roughness = (np.mean(np.abs(np.diff(raw.astype(float), axis=0))) +
                 np.mean(np.abs(np.diff(raw.astype(float), axis=1)))) / span
    median_error = abs(math.log(max(float(np.median(raw)), 1) / max(metadata["RawValueMedian"], 1)))
    return 100 * (1 - float(plausible.mean())) + roughness + 0.2 * median_error


def _decode_jpeg_ls(payload, width, height):
    # Some RGBT-Scenes captures use 16-bit JPEG-LS rather than PNG. Pillow's
    # baseline JPEG decoder cannot read them. Prefer CharLS via imagecodecs;
    # FFmpeg provides an alternative without changing the training environment.
    try:
        import imagecodecs
    except ImportError:
        imagecodecs = None
    if imagecodecs is not None:
        raw = np.asarray(imagecodecs.jpegls_decode(payload))
        if raw.dtype != np.uint16:
            raise ValueError("FLIR JPEG-LS must contain 16-bit scalar samples")
        return raw
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("This raw JPEG contains JPEG-LS. Install imagecodecs in the "
                           "conversion Python environment, or make FFmpeg available on PATH. "
                           "No additional codec is needed for thermal training after conversion.")
    result = subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "1",
        "-f", "image2pipe", "-vcodec", "jpegls", "-i", "pipe:0",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray16le", "pipe:1",
    ], input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
    if result.returncode or len(result.stdout) != width * height * 2:
        raise ValueError("FFmpeg could not decode the 16-bit FLIR JPEG-LS payload: " +
                         result.stderr.decode(errors="replace")[:300])
    return np.frombuffer(result.stdout, dtype="<u2").reshape(height, width).copy()


def decode_flir_rjpeg(path, byte_order="auto"):
    if byte_order not in ("auto", "native", "swap"):
        raise ValueError("byte_order must be auto, native, or swap")
    records = _records(_fff_from_jpeg(Path(path).read_bytes()))
    metadata = _camera_metadata(records[0x20])
    record = records[1]
    if len(record) < 32:
        raise ValueError("Truncated FLIR raw-data header")
    endian = _record_endian(record)
    width, height = struct.unpack_from(endian + "HH", record, 2)
    if min(width, height) < 2 or width * height > 100000000:
        raise ValueError("Invalid FLIR raw image dimensions")
    payload = record[32:]
    png = payload.startswith(b"\x89PNG\r\n\x1a\n")
    jpeg_ls = payload.startswith(b"\xff\xd8\xff\xf7")
    if png:
        if len(payload) < 26 or payload[24:26] != b"\x10\x00":
            raise ValueError("FLIR raw PNG must be 16-bit monochrome")
        with Image.open(io.BytesIO(payload)) as image:
            raw = np.asarray(image).astype(np.uint16)
    elif jpeg_ls:
        raw = _decode_jpeg_ls(payload, width, height)
    else:
        if len(payload) != width * height * 2:
            raise ValueError("Unsupported FLIR raw-data encoding")
        raw = np.frombuffer(payload, dtype=endian + "u2").reshape(height, width).astype(np.uint16)
    if raw.shape != (height, width):
        raise ValueError("FLIR raw dimensions do not match the header")
    native_score, swap_score = _byte_order_score(raw, metadata), _byte_order_score(raw.byteswap(), metadata)
    if byte_order == "auto":
        # Plain binary records use their declared endianness. PNG cameras may
        # violate PNG byte order; fail rather than guess when evidence is weak.
        if png and abs(native_score - swap_score) < 0.1:
            raise ValueError("Ambiguous FLIR PNG byte order; specify --byte_order native or swap")
        swapped = png and swap_score < native_score
    else:
        swapped = byte_order == "swap"
    if swapped:
        raw = raw.byteswap()
    signal = raw.astype(np.float64) + metadata["PlanckO"]
    if not np.isfinite(signal_to_apparent_temperature(signal, metadata)).all():
        raise ValueError("Invalid raw signal after byte-order correction")
    return raw.copy(), metadata, {"byteswap_used": bool(swapped),
                                 "raw_encoding": "PNG16" if png else ("JPEG-LS16" if jpeg_ls else "binary16"),
                                 "native_byte_order_score": native_score,
                                 "swapped_byte_order_score": swap_score}


def resize_signal(signal, size):
    """Resize in the linear signal domain; never resize temperature or uint8."""
    image = Image.fromarray(np.asarray(signal, dtype=np.float32))
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(image.resize(tuple(size), bilinear), dtype=np.float32).copy()


def load_signal_frame(root, record):
    root = Path(root).resolve()
    path = (root / record["signal_file"]).resolve()
    if root not in path.parents:
        raise ValueError("Radiometric manifest signal path escapes its directory")
    if file_sha256(path) != record["signal_sha256"]:
        raise ValueError("Radiometric signal content changed: " + str(path))
    signal = np.load(path, allow_pickle=False)
    if signal.dtype != np.float32 or signal.shape != tuple(record["shape"]):
        raise ValueError("Radiometric signal dtype/shape differs from its manifest")
    if not np.isfinite(signal).all():
        raise ValueError("Non-finite radiometric signal: " + str(path))
    return signal
