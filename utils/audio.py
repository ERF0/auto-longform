"""Audio utilities for warm, conversational explainer narration."""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import mimetypes
import random
import struct
import threading
import time
import wave
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import requests
from moviepy.audio.AudioClip import AudioArrayClip, AudioClip, CompositeAudioClip
from moviepy.audio.fx.all import audio_fadein, audio_fadeout, audio_loop, volumex
from moviepy.audio.io.AudioFileClip import AudioFileClip

from config import (
    BACKGROUND_SONGS_PATH,
    GEMINI_API_KEY,
    GEMINI_TTS_MAX_RETRIES,
    GEMINI_TTS_MODEL,
    GEMINI_TTS_RETRY_BASE_DELAY,
    GEMINI_TTS_RETRY_MAX_DELAY,
    GEMINI_TTS_THROTTLE_SECONDS,
    TEMP_PATH,
)

try:
    from pydub import AudioSegment
except ImportError:  # pragma: no cover - optional dependency
    AudioSegment = None

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class GeminiRateLimiter:
    """Thread-safe helper to coordinate Gemini TTS rate limits."""

    def __init__(self) -> None:
        self._next_allowed = 0.0
        self._quota_reset = 0.0
        self._lock = threading.Lock()

    def wait_for_window(self) -> None:
        delay = 0.0
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                delay = self._next_allowed - now
        if delay > 0:
            logger.debug("Waiting %.2fs before Gemini TTS request due to cooldown window.", delay)
            time.sleep(delay)

    def wait_for_quota(self) -> None:
        delay = 0.0
        with self._lock:
            now = time.monotonic()
            if now < self._quota_reset:
                delay = self._quota_reset - now
        if delay > 0:
            logger.warning(
                "Gemini TTS quota cooldown in effect. Waiting %.2fs before next request.", delay
            )
            time.sleep(delay)

    def schedule_next(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        with self._lock:
            self._next_allowed = max(self._next_allowed, time.monotonic()) + delay_seconds

    def note_quota_reset(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        with self._lock:
            self._quota_reset = max(self._quota_reset, time.monotonic() + delay_seconds)


_RATE_LIMITER = GeminiRateLimiter()

TARGET_DBFS = -2.0  # keep enough headroom for upbeat music beds while keeping narration forward
BED_FADE = 3.25
AMBIENT_FADE = 2.25
TEXTURE_VOLUME = 0.08
BED_VOLUME = 0.32
MAX_TEXTURE_LAYERS = 1
_MUSIC_WARNING_EMITTED = False


def save_binary_file(file_path: Path, data: bytes) -> None:
    file_path.write_bytes(data)


def _normalize_pcm16(wav_bytes: bytes) -> bytes:
    """Peak-normalize 16-bit PCM WAV to ~ -1 dBFS without clipping."""
    if len(wav_bytes) <= 44:
        return wav_bytes
    header, data = wav_bytes[:44], wav_bytes[44:]
    arr = np.frombuffer(data, dtype="<i2").astype(np.float32)
    peak = np.max(np.abs(arr)) if arr.size else 1.0
    if peak < 1e-6:
        return wav_bytes
    target = 0.89 * 32767.0
    gain = target / peak
    arr = np.clip(arr * gain, -32768, 32767).astype("<i2")
    return header + arr.tobytes()


def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    """Parse bits per sample and rate from an audio MIME type string."""
    bits_per_sample = 16
    rate = 24000
    channels = 1
    if not mime_type:
        return {"bits_per_sample": bits_per_sample, "rate": rate, "channels": channels}
    parts = mime_type.split(";")

    for param in parts[1:]:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate = int(param.split("=", 1)[1])
            except (ValueError, IndexError):
                continue
        elif param.lower().startswith("channels="):
            try:
                channels = int(param.split("=", 1)[1])
            except (ValueError, IndexError):
                continue
        elif "l16" in param.lower():
            bits_per_sample = 16
    return {"bits_per_sample": bits_per_sample, "rate": rate, "channels": channels}


def _pcm_bytes_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    params = parse_audio_mime_type(mime_type)
    bits_per_sample = params["bits_per_sample"]
    rate = params["rate"]
    channels = params.get("channels", 1)
    bytes_per_sample = max(1, bits_per_sample // 8)
    block_align = channels * bytes_per_sample
    byte_rate = rate * block_align
    chunk_size = 36 + len(audio_data)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        len(audio_data),
    )
    wav_bytes = header + audio_data
    try:
        return _normalize_pcm16(wav_bytes)
    except Exception as exc:
        logger.warning("Audio normalization failed: %s", exc)
        return wav_bytes


def convert_to_wav(audio_data: bytes, mime_type: str, speed: float = 1.0) -> bytes:
    """Convert arbitrary audio bytes to PCM WAV while optionally resampling."""
    subtype = ""
    if mime_type:
        subtype = mime_type.split("/", 1)[-1]
        subtype = subtype.split(";", 1)[0].lower()

    if subtype in {"wav", "wave", "x-wav"} or mimetypes.guess_type("x." + subtype)[0] == "audio/wav":
        try:
            return _normalize_pcm16(audio_data)
        except Exception as exc:
            logger.warning("Audio normalization failed: %s", exc)
            return audio_data

    if subtype in {"l16", "pcm", "x-pcm"} or mime_type.lower().startswith("audio/l16"):
        return _pcm_bytes_to_wav(audio_data, mime_type or "audio/L16")

    if AudioSegment is None:
        raise RuntimeError(
            f"Decoding {mime_type or 'unknown audio'} requires pydub. Install it via 'pip install pydub'."
        )

    try:
        segment = AudioSegment.from_file(io.BytesIO(audio_data), format=subtype or None)
    except Exception as exc:  # pragma: no cover - depends on codec availability
        raise RuntimeError(f"Unable to decode audio payload ({mime_type}): {exc}") from exc

    if speed and speed != 1.0:
        segment = segment._spawn(segment.raw_data, overrides={"frame_rate": int(segment.frame_rate * speed)})
        segment = segment.set_frame_rate(segment.frame_rate)

    segment = segment.set_channels(1).set_frame_rate(24000)
    buffer = io.BytesIO()
    segment.export(buffer, format="wav")
    wav_bytes = buffer.getvalue()
    try:
        wav_bytes = _normalize_pcm16(wav_bytes)
    except Exception as exc:  # pragma: no cover - best-effort normalization
        logger.warning("Audio normalization failed: %s", exc)
    return wav_bytes


def _calculate_retry_delay(attempt: int, response: Optional[requests.Response] = None) -> float:
    retry_after_header = response.headers.get("Retry-After") if response is not None else None
    delay = None
    if retry_after_header:
        try:
            delay = float(retry_after_header)
        except ValueError:
            try:
                retry_after_dt = parsedate_to_datetime(retry_after_header)
                now = datetime.now(timezone.utc)
                delay = max(0.0, (retry_after_dt - now).total_seconds())
            except (TypeError, ValueError, OverflowError):
                delay = None

    if delay is None:
        exponential = GEMINI_TTS_RETRY_BASE_DELAY * (2 ** (attempt - 1))
        delay = min(exponential, GEMINI_TTS_RETRY_MAX_DELAY)

    jitter = random.uniform(0, 0.25 * GEMINI_TTS_RETRY_BASE_DELAY)
    return min(delay + jitter, GEMINI_TTS_RETRY_MAX_DELAY)


def _parse_retry_info_delay(resp: Optional[requests.Response]) -> Optional[float]:
    if resp is None:
        return None
    try:
        payload = resp.json()
    except Exception:
        return None

    details = payload.get("error", {}).get("details", [])
    for detail in details:
        if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
            retry_delay = detail.get("retryDelay")
            if isinstance(retry_delay, str) and retry_delay.endswith("s"):
                try:
                    return float(retry_delay.rstrip("s"))
                except ValueError:
                    continue
    return None


def generate_voiceover(text: str) -> Path:
    """Generate a friendly, conversational narration track via Gemini TTS."""
    audio_id = uuid.uuid4()
    audio_path = TEMP_PATH / f"{audio_id}.wav"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "temperature": 0.65,
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "pulcherrima"}}},
        },
    }

    def _error_detail(resp: Optional[requests.Response]) -> str:
        if resp is None:
            return ""
        try:
            body = resp.json()
            return json.dumps(body, ensure_ascii=False)
        except Exception:
            try:
                return resp.text[:500]
            except Exception:
                return ""

    response: Optional[requests.Response] = None
    last_error: Optional[Exception] = None
    effective_max_attempts = max(2, GEMINI_TTS_MAX_RETRIES)
    attempt = 1
    while attempt <= effective_max_attempts:
        try:
            _RATE_LIMITER.wait_for_quota()
            _RATE_LIMITER.wait_for_window()
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(15, 240),  # generous connect/read timeouts for long synth tasks
            )
            response.raise_for_status()
            if GEMINI_TTS_THROTTLE_SECONDS > 0:
                _RATE_LIMITER.schedule_next(GEMINI_TTS_THROTTLE_SECONDS)
            break
        except requests.exceptions.HTTPError as err:
            status_code = err.response.status_code if err.response else None
            retryable = status_code in RETRYABLE_STATUS_CODES if status_code else False
            retry_delay = _parse_retry_info_delay(err.response)
            if retryable and attempt < effective_max_attempts:
                delay = retry_delay or _calculate_retry_delay(attempt, err.response)
                if retry_delay:
                    _RATE_LIMITER.note_quota_reset(retry_delay)
                _RATE_LIMITER.schedule_next(delay)
                logger.warning(
                    "Gemini TTS returned %s. Retrying in %.1fs (%s/%s). Details: %s",
                    status_code,
                    delay,
                    attempt,
                    effective_max_attempts,
                    _error_detail(err.response),
                )
                time.sleep(delay)
                attempt += 1
                continue
            if status_code == 429:
                delay_hint = retry_delay or _calculate_retry_delay(attempt, err.response)
                _RATE_LIMITER.note_quota_reset(delay_hint)
                raise Exception(
                    "Gemini TTS quota exceeded. Please wait before retrying or review billing limits."
                ) from err
            last_error = err
            detail = _error_detail(err.response)
            message = f"Gemini TTS API failed after {attempt} attempt(s): {err}"
            if detail:
                message = f"{message} | Detail: {detail}"
            raise Exception(message) from err
        except requests.exceptions.RequestException as err:
            if attempt < effective_max_attempts:
                delay = _calculate_retry_delay(attempt)
                _RATE_LIMITER.schedule_next(delay)
                logger.warning(
                    "Gemini TTS request error '%s'. Retrying in %.1fs (%s/%s)",
                    err,
                    delay,
                    attempt,
                    effective_max_attempts,
                )
                time.sleep(delay)
                attempt += 1
                continue
            last_error = err
            raise Exception(f"Gemini TTS API request failed: {err}") from err
        attempt += 1

    if response is None:
        if last_error:
            raise Exception(f"Gemini TTS API did not return a response (last error: {last_error})")
        raise Exception("Gemini TTS API did not return a response")

    response_data = response.json()
    candidates = response_data.get("candidates", [])
    if (
        not candidates
        or "content" not in candidates[0]
        or "parts" not in candidates[0]["content"]
        or not candidates[0]["content"]["parts"]
    ):
        raise Exception("Invalid response structure from Gemini TTS API")

    part = candidates[0]["content"]["parts"][0]
    inline_data = part.get("inlineData")
    if not inline_data or "data" not in inline_data:
        raise Exception("No audio data found in API response")

    audio_data = base64.b64decode(inline_data["data"])
    mime_type = inline_data.get("mimeType", "audio/wav")

    if mimetypes.guess_extension(mime_type) != ".wav":
        audio_data = convert_to_wav(audio_data, mime_type)
    else:
        try:
            audio_data = _normalize_pcm16(audio_data)
        except Exception as exc:
            logger.warning("Normalization failed on WAV: %s", exc)

    save_binary_file(audio_path, audio_data)
    return audio_path


def get_random_background_song() -> Optional[Path]:
    global _MUSIC_WARNING_EMITTED
    songs = list(BACKGROUND_SONGS_PATH.glob("*.mp3")) + list(BACKGROUND_SONGS_PATH.glob("*.wav"))
    if songs:
        return random.choice(songs)
    if not _MUSIC_WARNING_EMITTED:
        logger.warning("No background songs found in %s; continuing without music beds.", BACKGROUND_SONGS_PATH)
        _MUSIC_WARNING_EMITTED = True
    return None


def _list_audio_sources(directory: Path) -> List[Path]:
    return [path for path in directory.glob("*") if path.suffix.lower() in {".mp3", ".wav", ".aif", ".aiff"}]


def _ensure_audio_duration(clip: AudioFileClip, duration: float) -> AudioFileClip:
    if clip.duration >= duration:
        return clip.subclip(0, duration)
    looped = audio_loop(clip, duration=duration + BED_FADE)
    return looped.subclip(0, duration)


def _analyze_voice_energy(voiceover: AudioFileClip, window: float = 0.35) -> np.ndarray:
    samples = voiceover.to_soundarray(fps=int(1 / window))
    energy = np.linalg.norm(samples, axis=1)
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    return energy


def _ducking_curve(energy: np.ndarray, duration: float):
    if energy.size == 0:
        return lambda t: 1.0
    smooth_energy = energy
    if energy.size > 4:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        smooth_energy = np.convolve(energy, kernel, mode="same")
        smooth_energy = np.clip(smooth_energy, 0.0, 1.0)
    times = np.linspace(0, duration, num=energy.size)

    def _curve(t: float) -> float:
        t = max(0.0, min(duration, t))
        idx = np.searchsorted(times, t)
        idx = np.clip(idx, 0, energy.size - 1)
        energy_level = smooth_energy[idx]
        return float(0.3 + 0.55 * (1.0 - energy_level))

    return _curve


def _normalize_audio_clip(clip: AudioClip, target_dbfs: float = TARGET_DBFS) -> AudioClip:
    try:
        if getattr(clip, "duration", None) is None:
            raise ValueError("duration missing")
        array = clip.to_soundarray(fps=44100, nbytes=2)
    except Exception as exc:
        logger.warning("Skipping audio normalization; failed to read sound array: %s", exc)
        return clip
    peak = np.max(np.abs(array))
    if peak <= 0:
        return clip
    target_amp = 10 ** (target_dbfs / 20)
    gain = target_amp / peak
    return clip.volumex(gain)


def _soft_compress_voiceover(
    clip: AudioFileClip,
    threshold_db: float = -14.0,
    ratio: float = 2.2,
    makeup_db: float = 3.0,
    fps: int = 44100,
) -> AudioClip:
    """Apply gentle compression and gain to make narration feel closer and warmer."""
    try:
        array = clip.to_soundarray(fps=fps).astype(np.float32)
    except Exception as exc:
        logger.warning("Voiceover enhancement skipped; failed to read samples: %s", exc)
        return clip
    if not isinstance(array, np.ndarray) or array.size == 0:
        logger.warning("Voiceover enhancement skipped; empty or invalid sample array.")
        return clip

    array = np.clip(array, -1.0, 1.0)
    amplitude = np.maximum(np.abs(array), 1e-6)
    db = 20.0 * np.log10(amplitude)
    compressed_db = np.where(db > threshold_db, threshold_db + (db - threshold_db) / ratio, db)
    compressed = np.sign(array) * (10.0 ** (compressed_db / 20.0))
    compressed *= 10.0 ** (makeup_db / 20.0)
    compressed = np.clip(compressed, -1.0, 1.0)

    enhanced = AudioArrayClip(compressed, fps=fps)
    enhanced = audio_fadein(audio_fadeout(enhanced, 0.25), 0.12)
    return enhanced


def _humanize_voice_clip(clip: AudioClip) -> AudioClip:
    """Make narration feel closer via gentle compression and consistent loudness."""
    duration = getattr(clip, "duration", None)
    try:
        base = clip
        if duration:
            base = base.set_duration(duration)
        processed = _soft_compress_voiceover(base)
        processed_duration = duration or getattr(processed, "duration", None)
        if processed_duration:
            processed = processed.set_duration(processed_duration)
        return processed.set_fps(44100).volumex(1.05)
    except Exception as exc:
        logger.warning("Voice humanization skipped; using raw clip: %s", exc)
        fallback = clip
        if duration:
            fallback = fallback.set_duration(duration)
        return fallback.set_fps(44100).volumex(1.05)


def _read_wav_array(path: Path) -> tuple[Optional[np.ndarray], int]:
    """Load PCM WAV samples with minimal dependencies."""
    try:
        with wave.open(str(path), "rb") as wav:
            sample_width = wav.getsampwidth()
            channels = wav.getnchannels()
            rate = wav.getframerate() or 24000
            frames = wav.readframes(wav.getnframes())
        if not frames or sample_width not in (1, 2, 4):
            return None, rate
        dtype = {1: "<i1", 2: "<i2", 4: "<i4"}[sample_width]
        array = np.frombuffer(frames, dtype=dtype)
        if channels > 1:
            array = array.reshape(-1, channels)
        else:
            array = array.reshape(-1, 1)
        max_val = float(2 ** (8 * sample_width - 1))
        array = array.astype(np.float32) / max_val
        return array, rate
    except Exception as exc:
        logger.debug("Wave decode failed for %s: %s", path, exc)
        return None, 24000


def _load_voiceover_clip(path: Path) -> AudioClip:
    """Prefer raw PCM loading to avoid flaky readers."""
    array, rate = _read_wav_array(path)
    if array is not None and array.size:
        return AudioArrayClip(array, fps=rate)
    alt = _segment_array_from_file(path)
    if alt is not None and alt[0].size:
        return AudioArrayClip(alt[0], fps=alt[1])
    clip = AudioFileClip(str(path))
    if getattr(clip, "reader", None) is None:
        raise RuntimeError("Voiceover file could not be decoded (no audio reader).")
    return clip


def _segment_array_from_file(path: Path) -> Optional[tuple[np.ndarray, int]]:
    """Decode via pydub for stubborn files."""
    if AudioSegment is None:
        return None
    try:
        segment = AudioSegment.from_file(str(path))
        segment = segment.set_channels(1).set_frame_rate(24000)
        samples = np.array(segment.get_array_of_samples())
        if segment.channels > 1:
            samples = samples.reshape((-1, segment.channels))
        else:
            samples = samples.reshape((-1, 1))
        max_val = float(1 << (8 * segment.sample_width - 1))
        samples = samples.astype(np.float32) / max_val
        return samples, int(segment.frame_rate)
    except Exception as exc:
        logger.debug("pydub decode failed for %s: %s", path, exc)
        return None


def _collect_audio_array(path: Path, target_rate: int = 24000) -> tuple[np.ndarray, int]:
    """Decode audio into a numpy array with multiple fallbacks."""
    array, rate = _read_wav_array(path)
    if array is not None and array.size:
        return array.astype(np.float32), rate

    fallback = _segment_array_from_file(path)
    if fallback is not None and fallback[0].size:
        return fallback[0].astype(np.float32), fallback[1]

    clip = None
    try:
        clip = AudioFileClip(str(path))
        arrays = list(clip.iter_chunks(fps=target_rate, quantize=True))
        if not arrays:
            raise RuntimeError("AudioFileClip returned no chunks")
        stacked = np.concatenate(arrays, axis=0)
        return stacked.astype(np.float32), target_rate
    finally:
        with suppress(Exception):
            if clip is not None:
                clip.close()


def _ensure_mono(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2 and array.shape[1] == 1:
        return array
    # average channels to mono to avoid shape issues downstream
    return np.mean(array, axis=1, keepdims=True)


def _compress_voiceover_array(
    array: np.ndarray,
    fps: int,
    threshold_db: float = -14.0,
    ratio: float = 2.2,
    makeup_db: float = 3.0,
) -> AudioArrayClip:
    """Apply gentle compression directly on raw samples."""
    if not isinstance(array, np.ndarray) or array.size == 0:
        raise RuntimeError("Voiceover audio is empty.")
    array = _ensure_mono(array)
    array = np.clip(array.astype(np.float32), -1.0, 1.0)
    amplitude = np.maximum(np.abs(array), 1e-6)
    db = 20.0 * np.log10(amplitude)
    compressed_db = np.where(db > threshold_db, threshold_db + (db - threshold_db) / ratio, db)
    compressed = np.sign(array) * (10.0 ** (compressed_db / 20.0))
    compressed *= 10.0 ** (makeup_db / 20.0)
    compressed = np.clip(compressed, -1.0, 1.0)
    clip = AudioArrayClip(compressed, fps=fps)
    clip = audio_fadein(audio_fadeout(clip, 0.25), 0.12)
    return clip


def _optional_texture_layers(duration: float) -> List[AudioFileClip]:
    textures_dir = BACKGROUND_SONGS_PATH / "textures"
    if not textures_dir.exists():
        return []
    layers: List[AudioFileClip] = []
    for texture_path in _list_audio_sources(textures_dir)[:MAX_TEXTURE_LAYERS]:
        try:
            texture = AudioFileClip(str(texture_path))
            texture = _ensure_audio_duration(texture, duration)
            texture = audio_fadein(audio_fadeout(texture, AMBIENT_FADE), AMBIENT_FADE)
            texture = texture.volumex(TEXTURE_VOLUME)
            layers.append(texture)
        except Exception as exc:
            logger.warning("Failed to load ambient layer %s: %s", texture_path, exc)
    return layers


def create_conversational_soundtrack(voiceover_path: Path) -> AudioClip:
    """Blend voiceover, upbeat beds, and light textures for conversational energy."""
    voiceover = None
    duration = 0.0
    direct = None
    used_direct = False

    # First try to use the raw file directly to avoid resample/pitch surprises.
    try:
        direct = AudioFileClip(str(voiceover_path))
        duration = direct.duration or 0.0
        if duration <= 0:
            raise RuntimeError("voiceover duration missing")
        voiceover = _humanize_voice_clip(direct.set_duration(duration))
        used_direct = True
    except Exception:
        pass
    finally:
        with suppress(Exception):
            if direct is not None and not used_direct:
                direct.close()

    if voiceover is None:
        raw_array, raw_rate = _collect_audio_array(voiceover_path)
        voiceover = _compress_voiceover_array(raw_array, raw_rate)
        duration = voiceover.duration
        voiceover = _humanize_voice_clip(voiceover.set_duration(duration))

    soundtrack_layers: List[AudioClip] = [voiceover]

    bed_path = get_random_background_song()
    if bed_path:
        try:
            bed = AudioFileClip(str(bed_path))
            bed = _ensure_audio_duration(bed, duration)
            bed = audio_fadein(audio_fadeout(bed, BED_FADE), BED_FADE)
            energy = _analyze_voice_energy(voiceover)
            duck_curve = _ducking_curve(energy, duration)
            bed = bed.set_fps(44100).volumex(lambda t: (BED_VOLUME - 0.05) * duck_curve(t))
            soundtrack_layers.append(bed)
        except Exception as exc:
            logger.warning("Failed to use background score %s: %s", bed_path, exc)

    soundtrack_layers.extend(_optional_texture_layers(duration))

    composite = CompositeAudioClip(soundtrack_layers).set_duration(duration).set_fps(44100)
    return _normalize_audio_clip(composite, target_dbfs=TARGET_DBFS)
