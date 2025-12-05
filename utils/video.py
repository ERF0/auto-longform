"""High-end short-form video renderer with cinematic pacing and captions."""

from __future__ import annotations

import gc
import logging
import random
import re
import math
import uuid
from contextlib import suppress
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import assemblyai as aai
import numpy as np
import requests
from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    VideoFileClip,
)
from moviepy.video.fx import all as vfx
from moviepy.video.fx.all import crop, fadein, fadeout, loop

try:
    from PIL import Image, ImageColor, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Pillow is required for captions
    Image = None
    ImageColor = None
    ImageDraw = None
    ImageFont = None

from config import (
    ASSEMBLY_AI_API_KEY,
    CANVAS_SIZE as CONFIG_CANVAS_SIZE,
    CANVAS_TINT,
    CAPTION_BACKGROUND,
    CAPTION_BG_OPACITY,
    CAPTION_COLOR,
    CAPTION_FADE,
    CAPTION_FONT_PATH,
    CAPTION_FONT_SIZE,
    CAPTION_MAX_WIDTH_RATIO,
    CAPTION_PADDING,
    CAPTION_STROKE,
    DISABLE_SUBTITLES,
    FILM_GRAIN_OPACITY,
    LETTERBOX_HEIGHT,
    LOW_MEMORY_MODE,
    OUTPUT_PATH,
    SAFE_MODE,
    SECONDARY_CONTENT_PATH,
    SEGMENT_DURATION_RANGE,
    TEMP_PATH,
    TARGET_FPS,
    TRANSITION_DURATION,
    THREADS as CONFIG_THREADS,
    VISUAL_SEGMENT_RANGE,
    VIDEO_STYLE,
    STOCK_FOOTAGE_CONFIG,
)
from utils.audio import create_conversational_soundtrack

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
CANVAS_SIZE = (max(CONFIG_CANVAS_SIZE[0], 1920), max(CONFIG_CANVAS_SIZE[1], 1080))
CAPTION_MAX_WIDTH = int(CANVAS_SIZE[0] * CAPTION_MAX_WIDTH_RATIO)
THREADS = CONFIG_THREADS


@dataclass(frozen=True)
class VideoSettings:
    canvas_size: Tuple[int, int] = CANVAS_SIZE
    transition: float = TRANSITION_DURATION
    visual_segment_range: Tuple[float, float] = VISUAL_SEGMENT_RANGE
    min_segment_step: float = SEGMENT_DURATION_RANGE[0]
    target_fps: int = TARGET_FPS
    grain_opacity: float = FILM_GRAIN_OPACITY
    letterbox_height: int = LETTERBOX_HEIGHT


SETTINGS = VideoSettings()

ENABLE_SEMANTIC_MATCHING = bool(STOCK_FOOTAGE_CONFIG.get("enable_semantic_matching", True))
OVERLAY_FREQUENCY = float(STOCK_FOOTAGE_CONFIG.get("overlay_frequency", 0.6))
SPLIT_SCREEN_FREQUENCY = float(STOCK_FOOTAGE_CONFIG.get("split_screen_frequency", 0.15))
SEGMENT_DURATION_MULTIPLIER = float(STOCK_FOOTAGE_CONFIG.get("segment_duration_multiplier", 1.0))


def _resolve_assets(video_paths: Sequence[Path]) -> List[Path]:
    candidates: List[Path] = []
    for raw in video_paths:
        path = Path(raw)
        if not path.exists():
            logger.warning("Asset does not exist: %s", path)
            continue
        if not path.is_file():
            logger.warning("Asset is not a file: %s", path)
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            logger.warning("Could not stat asset %s: %s", path, exc)
            continue
        if size <= 1024:
            logger.warning("Skipping suspiciously small asset (%d bytes): %s", size, path)
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            logger.warning("Unsupported asset extension %s for %s", path.suffix, path)
            continue
        candidates.append(path)

    if candidates:
        logger.info("Resolved %d valid assets from %d provided paths", len(candidates), len(video_paths))
        return candidates

    if not SECONDARY_CONTENT_PATH.exists():
        return []
    fallback = [
        path
        for path in sorted(SECONDARY_CONTENT_PATH.glob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and path.stat().st_size > 1024
    ]
    if fallback:
        logger.info("Using %d secondary assets as fallback", len(fallback))
    return fallback


def _resize_to_canvas(clip: VideoClip, canvas_size: Tuple[int, int]) -> VideoClip:
    if clip.w == canvas_size[0] and clip.h == canvas_size[1]:
        return clip
    width_ratio = canvas_size[0] / clip.w
    height_ratio = canvas_size[1] / clip.h
    scale = max(width_ratio, height_ratio) * 1.02
    resized = clip.resize(scale)
    x_center = resized.w / 2
    y_center = resized.h / 2
    return crop(
        resized,
        width=canvas_size[0],
        height=canvas_size[1],
        x_center=x_center,
        y_center=y_center,
    )


def _ensure_duration(clip: VideoClip, duration: float, transition: float) -> VideoClip:
    if clip.duration >= duration:
        return clip.subclip(0, duration)
    return loop(clip, duration=duration + transition).subclip(0, duration)


def _apply_motion(clip: VideoClip, duration: float) -> VideoClip:
    if VIDEO_STYLE == "investigative":
        zoom_strength = random.uniform(0.005, 0.015)
    elif VIDEO_STYLE == "cinematic":
        zoom_strength = random.uniform(0.01, 0.02)
    else:
        zoom_strength = random.uniform(0.008, 0.015)

    def _zoom(t: float) -> float:
        progress = min(max(t / max(duration, 0.01), 0.0), 1.0)
        return 1.0 + zoom_strength * progress

    animated = clip.fx(vfx.resize, _zoom)
    return animated


def _apply_visual_enhancements(clip: VideoClip) -> VideoClip:
    enhanced = clip.fx(vfx.colorx, 1.04)
    enhanced = enhanced.fx(vfx.lum_contrast, lum=0.01, contrast=1.03)
    enhanced = enhanced.fx(vfx.gamma_corr, 0.98)
    return enhanced


def _load_asset_clip(
    path: Path,
    settings: VideoSettings,
    cache: dict[Path, VideoClip],
) -> VideoClip:
    """Load a visual asset once and reuse the reader across segments."""
    cached = cache.get(path)
    if cached is not None:
        return cached
    try:
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            clip = VideoFileClip(
                str(path),
                target_resolution=(settings.canvas_size[1], settings.canvas_size[0]),
                fps_source="tbr",
            ).without_audio()
        else:
            clip = ImageClip(str(path)).set_duration(max(settings.visual_segment_range[0], 3.0))
    except Exception as exc:
        raise RuntimeError(f"Failed to load asset {path}: {exc}") from exc
    clip = _resize_to_canvas(clip, settings.canvas_size)
    cache[path] = clip
    return clip


def _segment_from_asset(base_clip: VideoClip, duration: float, settings: VideoSettings) -> VideoClip:
    """Create a timeline segment from a cached asset without spawning new readers."""
    try:
        if isinstance(base_clip, VideoFileClip):
            base_duration = base_clip.duration or duration
            subclip_duration = min(base_duration, duration + settings.transition)
            if subclip_duration < 0.5:
                raise RuntimeError(f"Asset too short for segment: {subclip_duration:.2f}s")
            max_start = max(0.0, base_duration - subclip_duration)
            start_time = random.uniform(0.0, max_start)
            working = base_clip.subclip(start_time, start_time + subclip_duration)
        else:
            working = base_clip.copy().set_duration(duration)
    except Exception as exc:
        raise RuntimeError(f"Failed to prepare visual segment: {exc}") from exc
    working = _ensure_duration(working, duration, settings.transition)
    working = _apply_motion(working, duration)
    working = _apply_visual_enhancements(working)
    return working


def _match_asset_to_segment(
    segment_text: str,
    available_assets: List[Path],
    search_mappings: List[Dict[str, Any]],
) -> Path | None:
    """Pick the most relevant asset based on keyword matching."""
    if not available_assets:
        return None
    if not search_mappings:
        return random.choice(available_assets)

    segment_lower = segment_text.lower()
    scored_assets: List[tuple[Path, int]] = []

    for asset_path in available_assets:
        score = 0
        for mapping in search_mappings:
            for term in mapping.get("terms", []):
                if isinstance(term, str) and term.lower() in segment_lower:
                    score += 1
        scored_assets.append((asset_path, score))

    scored_assets.sort(key=lambda item: item[1], reverse=True)

    if scored_assets and random.random() < 0.7 and scored_assets[0][1] > 0:
        return scored_assets[0][0]

    pool = [asset for asset, _ in scored_assets[:3] if asset.exists()] or available_assets
    return random.choice(pool) if pool else None


def _build_gradient_overlay(duration: float, canvas_size: Tuple[int, int]) -> VideoClip:
    """Create a dual-gradient overlay with subtle tinting for depth."""
    width, height = canvas_size
    gradient = np.zeros((height, width, 4), dtype=np.uint8)

    primary_alpha = (180 * (np.linspace(0, 1, height, dtype=np.float32) ** 1.6)).astype(np.uint8)
    y_coords, x_coords = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    radius = max(width, height) * 0.8
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    radial_alpha = (80 * (1 - np.clip(distances / radius, 0, 1) ** 2)).astype(np.uint8)

    combined_alpha = np.minimum(primary_alpha[:, None] + radial_alpha, 255)
    gradient[:, :, 3] = combined_alpha

    tint = random.choice([(8, 12, 25), (15, 8, 20), (5, 15, 30)])
    for idx, value in enumerate(tint):
        gradient[:, :, idx] = np.uint8(value)

    overlay = ImageClip(gradient).set_duration(duration)
    return overlay


def _create_animated_gradient_background(duration: float, settings: VideoSettings) -> VideoClip:
    """Stable, subtle gradient fallback when no stock assets are available."""
    width, height = settings.canvas_size
    base_color = np.array(CANVAS_TINT, dtype=np.uint8)

    def make_frame(t: float) -> np.ndarray:
        progress = t / max(duration, 0.001)
        vertical = np.linspace(0.97, 1.03, height, dtype=np.float32).reshape(height, 1, 1)
        hue_shift = 0.01 * math.sin(progress * 0.25)

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for idx, channel in enumerate(base_color):
            delta = hue_shift * (1 if idx == 0 else -0.5 if idx == 1 else 0.3)
            channel_val = channel * vertical * (1 + delta)
            frame[:, :, idx] = np.clip(channel_val, 0, 255).astype(np.uint8)
        return frame

    return VideoClip(make_frame=make_frame, duration=duration)


def _background_layers(duration: float, settings: VideoSettings) -> List[VideoClip]:
    base_color = np.array(CANVAS_TINT, dtype=np.uint8)
    base = ColorClip(size=settings.canvas_size, color=base_color).set_duration(duration)

    if LOW_MEMORY_MODE:
        layers = [base]
    else:
        gradient = _build_gradient_overlay(duration, settings.canvas_size)

        rng = np.random.default_rng()
        width, height = settings.canvas_size

        def _grain_frame(_: float) -> np.ndarray:
            noise = rng.normal(0.0, 18.0, size=(height, width, 1))
            noise = np.clip(128 + noise, 0, 255).astype(np.uint8)
            return np.repeat(noise, 3, axis=2)

        grain = VideoClip(make_frame=_grain_frame, duration=duration).set_opacity(settings.grain_opacity)
        layers = [base, gradient, grain]

    letterbox_color = np.array((0, 0, 0), dtype=np.uint8)
    letterbox = ColorClip(
        size=(settings.canvas_size[0], settings.letterbox_height), color=letterbox_color
    ).set_opacity(0.55)
    top = letterbox.set_duration(duration).set_position(("center", "top"))
    bottom = letterbox.set_duration(duration).set_position(("center", "bottom"))

    layers.extend([top, bottom])
    return layers


def _build_storyboard(
    asset_paths: Sequence[Path],
    total_duration: float,
    settings: VideoSettings,
    script_segments: List[str] | None = None,
    search_mappings: List[Dict[str, Any]] | None = None,
) -> tuple[List[VideoClip], List[VideoClip]]:
    assets = list(asset_paths)
    if not assets:
        return [_create_animated_gradient_background(total_duration, settings)], []

    script_segments = script_segments or []
    search_mappings = search_mappings or []

    clips: List[VideoClip] = []
    asset_cycle = cycle(assets)
    current_start = 0.0
    max_attempts = max(1, len(assets))
    broken_assets: set[Path] = set()
    asset_cache: dict[Path, VideoClip] = {}
    cached_handles: List[VideoClip] = []
    cache_ids: set[int] = set()
    overlay_chance = max(0.0, min(1.0, OVERLAY_FREQUENCY * 0.5))
    duration_multiplier = max(0.2, SEGMENT_DURATION_MULTIPLIER)

    def _fallback_clip(duration: float) -> VideoClip:
        color = np.array(CANVAS_TINT, dtype=np.uint8).reshape(1, 1, 3)
        height, width = settings.canvas_size[1], settings.canvas_size[0]
        for scale in (1.0, 0.75, 0.5):
            w = max(2, int(width * scale))
            h = max(2, int(height * scale))
            try:
                frame = np.empty((h, w, 3), dtype=np.uint8)
                frame[...] = color
                clip = VideoClip(make_frame=lambda _: frame, duration=duration).set_position("center")
                if (w, h) != (width, height):
                    logger.warning("Using reduced fallback resolution %dx%d (canvas %dx%d)", w, h, width, height)
                return clip
            except MemoryError:
                logger.warning("Fallback clip allocation failed at %dx%d; retrying smaller.", w, h)
        raise RuntimeError("Unable to allocate fallback clip due to memory limits.")

    while current_start < total_duration:
        base_duration = random.uniform(
            settings.visual_segment_range[0] * 0.7,
            settings.visual_segment_range[1] * 0.8,
        )
        base_duration *= duration_multiplier
        remaining = total_duration - current_start
        segment_duration = min(base_duration, remaining + settings.transition)

        current_segment = ""
        if script_segments:
            seg_idx = min(
                len(script_segments) - 1,
                int(current_start / max(total_duration, 0.001) * len(script_segments)),
            )
            current_segment = script_segments[seg_idx]

        clip: VideoClip | None = None
        used_asset = False
        asset_path: Path | None = None
        available_assets = [path for path in assets if path not in broken_assets]
        candidate_assets: List[Path] = []
        if ENABLE_SEMANTIC_MATCHING:
            matched_asset = _match_asset_to_segment(current_segment, available_assets, search_mappings)
            if matched_asset is not None:
                candidate_assets.append(matched_asset)

        for _ in range(len(available_assets)):
            candidate = next(asset_cycle)
            if candidate in broken_assets or candidate in candidate_assets:
                continue
            candidate_assets.append(candidate)
            if len(candidate_assets) >= max_attempts:
                break

        if not candidate_assets:
            candidate_assets = available_assets

        for asset_path in candidate_assets[:max_attempts]:
            if asset_path in broken_assets:
                continue
            try:
                base_clip = _load_asset_clip(asset_path, settings, asset_cache)
                if id(base_clip) not in cache_ids:
                    cache_ids.add(id(base_clip))
                    cached_handles.append(base_clip)
                clip = _segment_from_asset(base_clip, segment_duration, settings)
                used_asset = True
                break
            except Exception as exc:
                logger.warning("Skipping broken visual asset %s: %s", asset_path, exc)
                broken_assets.add(asset_path)

        if clip is None:
            gc.collect()
            if len(broken_assets) >= len(assets):
                remaining_duration = max(0.01, total_duration - current_start)
                logger.warning("All visual assets failed; using fallback canvas for remaining %.2fs", remaining_duration)
                clip = _create_animated_gradient_background(remaining_duration, settings)
                clips.append(clip.set_start(max(0.0, current_start - settings.transition)))
                break
            try:
                clip = _create_animated_gradient_background(segment_duration, settings)
            except Exception as fallback_exc:
                logger.error("Even fallback clip creation failed: %s", fallback_exc)
                raise

        clip_start = max(0.0, current_start - settings.transition)
        clip = clip.set_start(clip_start)
        clips.append(clip)

        if used_asset and len(assets) > 1 and asset_path is not None:
            overlay_candidates = [path for path in assets if path != asset_path and path not in broken_assets]
            if overlay_candidates and random.random() < overlay_chance:
                overlay_path = random.choice(overlay_candidates)
                try:
                    overlay_base = _load_asset_clip(overlay_path, settings, asset_cache)
                    if id(overlay_base) not in cache_ids:
                        cache_ids.add(id(overlay_base))
                        cached_handles.append(overlay_base)
                    overlay_clip = _segment_from_asset(
                        overlay_base, max(0.5, segment_duration * 0.7), settings
                    ).set_start(clip_start + segment_duration * 0.2)
                    overlay_clip = overlay_clip.resize(0.25).set_position(("right", "bottom"))
                    overlay_clip = overlay_clip.set_opacity(0.3)
                    clips.append(overlay_clip)
                except Exception as exc:
                    logger.debug("Skipping overlay: %s", exc)

        if len(clips) % 5 == 0:
            gc.collect()

        step = max(segment_duration * 0.82, settings.min_segment_step)
        current_start += step

    return clips, cached_handles


def _color_tuple(value: str | Tuple[int, int, int], alpha: int = 255) -> Tuple[int, int, int, int]:
    if isinstance(value, tuple):
        r, g, b = value[:3]
    elif ImageColor is not None:
        r, g, b = ImageColor.getrgb(value)
    else:  # pragma: no cover
        r, g, b = (255, 255, 255)
    return (int(r), int(g), int(b), alpha)


def _load_font(font_size: int):
    if ImageFont is None:
        raise RuntimeError("Pillow is required for caption rendering.")
    try:
        return ImageFont.truetype(CAPTION_FONT_PATH, font_size)
    except OSError:
        return ImageFont.load_default()


def _wrap_lines(text: str, font) -> List[str]:
    draw_canvas = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(draw_canvas)
    lines: List[str] = []
    current = ""

    def _split_word(word: str) -> List[str]:
        if draw.textlength(word, font=font) <= CAPTION_MAX_WIDTH:
            return [word]
        chunks: List[str] = []
        chunk = ""
        for char in word:
            candidate = f"{chunk}{char}"
            if draw.textlength(candidate, font=font) <= CAPTION_MAX_WIDTH:
                chunk = candidate
                continue
            if chunk:
                chunks.append(f"{chunk}-")
            chunk = char
        if chunk:
            chunks.append(chunk)
        return chunks

    for raw_word in text.split():
        for word in _split_word(raw_word):
            trial = f"{current} {word}".strip()
            width = draw.textlength(trial, font=font)
            if width <= CAPTION_MAX_WIDTH or not current:
                current = trial
                continue
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _render_caption(text: str) -> np.ndarray:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for caption rendering.")
    font = _load_font(max(12, CAPTION_FONT_SIZE - 4))
    lines = _wrap_lines(text, font)

    draw_canvas = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(draw_canvas)

    text_width = 0
    text_height = 0
    for line in lines:
        text_width = max(text_width, int(draw.textlength(line, font=font)))
        text_height += CAPTION_FONT_SIZE + 8

    text_height += CAPTION_PADDING * 2
    text_width = min(CAPTION_MAX_WIDTH + CAPTION_PADDING * 2, text_width + CAPTION_PADDING * 2)

    canvas = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    if CAPTION_BG_OPACITY > 0:
        draw.rectangle(
            [0, 0, text_width, text_height],
            fill=_color_tuple(CAPTION_BACKGROUND, int(255 * CAPTION_BG_OPACITY)),
        )

    shadow_offsets = [(-3, -3), (3, 3)]
    shadow_color = _color_tuple("#000000", 200)
    y = CAPTION_PADDING
    try:
        bbox = draw.textbbox((0, 0), "Hg", font=font)
        line_height = bbox[3] - bbox[1]
    except Exception:
        line_height = CAPTION_FONT_SIZE
    for line in lines:
        # Softer shadow for legibility on busy footage
        for dx, dy in shadow_offsets:
            draw.text(
                (CAPTION_PADDING + dx, y + dy),
                line,
                font=font,
                fill=shadow_color,
                stroke_width=0,
            )
            draw.text(
                (CAPTION_PADDING, y),
                line,
                font=font,
                fill=_color_tuple(CAPTION_COLOR, 255),
                stroke_width=1,
                stroke_fill=_color_tuple(CAPTION_STROKE, 255),
            )
        y += line_height + 8

    return np.array(canvas)


def generate_word_timestamps(audio_path: Path) -> List[Tuple[str, float, float]]:
    if not ASSEMBLY_AI_API_KEY:
        return []
    aai.settings.api_key = ASSEMBLY_AI_API_KEY
    transcript = aai.Transcriber().transcribe(str(audio_path))

    words: List[Tuple[str, float, float]] = []
    if transcript.words:
        for word in transcript.words:
            words.append((word.text, word.start / 1000.0, word.end / 1000.0))
    return words


def _approximate_segments_from_script(script: str, duration: float) -> List[Tuple[str, float, float]]:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", script) if segment.strip()]
    if not sentences:
        return []
    word_counts = [max(1, len(sentence.split())) for sentence in sentences]
    total_words = sum(word_counts)

    segments: List[Tuple[str, float, float]] = []
    cursor = 0.0
    for sentence, weight in zip(sentences, word_counts):
        portion = weight / total_words
        seg_duration = max(2.0, duration * portion)
        start = cursor
        end = min(duration, start + seg_duration)
        segments.append((sentence, start, end))
        cursor = end

    if segments:
        text, start, _ = segments[-1]
        segments[-1] = (text, start, duration)
    return segments


def _segments_from_words(words: List[Tuple[str, float, float]], duration: float) -> List[Tuple[str, float, float]]:
    if not words:
        return []
    segments: List[Tuple[str, float, float]] = []
    buffer: List[str] = []
    seg_start = words[0][1]

    for word, start, end in words:
        buffer.append(word)
        clause_end = word.endswith((".", "!", "?"))
        if len(buffer) >= 7 or clause_end or (end - seg_start) >= 4.5:
            text = " ".join(buffer).strip()
            segments.append((text, seg_start, min(duration, end)))
            buffer = []
            seg_start = min(duration, end)

    if buffer:
        text = " ".join(buffer).strip()
        segments.append((text, seg_start, duration))

    return segments


def _load_caption_segments(audio_path: Path, duration: float, script: Optional[str]) -> List[Tuple[str, float, float]]:
    try:
        words = generate_word_timestamps(audio_path)
    except Exception as exc:
        logger.warning("AssemblyAI subtitles failed: %s", exc)
        words = []

    if words:
        return _segments_from_words(words, duration)
    if script:
        return _approximate_segments_from_script(script, duration)
    return []


def _build_caption_layers(audio_path: Path, duration: float, script: Optional[str]) -> List[VideoClip]:
    segments = _load_caption_segments(audio_path, duration, script)
    if not segments:
        return []

    layers: List[VideoClip] = []
    safe_bottom = CANVAS_SIZE[1] - LETTERBOX_HEIGHT - 90

    for text, start, end in segments:
        bounded_start = max(0.0, min(start, duration))
        bounded_end = max(bounded_start + 0.3, min(end, duration))
        clip_duration = bounded_end - bounded_start
        if clip_duration <= 0.3:
            continue

        try:
            frame = _render_caption(text)
        except Exception as exc:
            logger.warning("Skipping caption '%s': %s", text, exc)
            continue

        base_x = (CANVAS_SIZE[0] - frame.shape[1]) / 2
        base_y = max(CAPTION_PADDING, safe_bottom - frame.shape[0])
        caption_clip = ImageClip(frame).set_duration(clip_duration)
        caption_clip = caption_clip.set_position((base_x, base_y))

        fade_amount = min(0.2, clip_duration * 0.3)
        caption_clip = fadein(fadeout(caption_clip, fade_amount), fade_amount)
        caption_clip = caption_clip.set_start(bounded_start)
        layers.append(caption_clip.set_opacity(0.96))

    return layers


def _close_media_resources(clips: Iterable[Optional[object]]) -> None:
    seen: set[int] = set()
    for clip in clips:
        if clip is None:
            continue
        identifier = id(clip)
        if identifier in seen:
            continue
        seen.add(identifier)
        close_method = getattr(clip, "close", None)
        if callable(close_method):
            with suppress(Exception):
                close_method()


def _add_final_polish(final_clip: CompositeVideoClip, duration: float, settings: VideoSettings) -> CompositeVideoClip:
    """Apply a light film grain and vignette to the composite video."""
    grain_intensity = max(0.0, min(0.02, settings.grain_opacity * 1.2))
    rng = np.random.default_rng(42)

    def _apply_grain(get_frame, t: float) -> np.ndarray:
        frame = get_frame(t)
        if grain_intensity <= 0:
            return frame
        noise = rng.normal(0.0, grain_intensity * 255, size=frame.shape).astype(np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    polished = final_clip.fl(_apply_grain)

    width, height = settings.canvas_size
    y_coords, x_coords = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2
    radius = max(width, height) * 0.8
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    vignette_alpha = (30 * np.clip(distances / radius, 0, 1) ** 1.8).astype(np.uint8)

    vignette = np.zeros((height, width, 4), dtype=np.uint8)
    vignette[:, :, 3] = vignette_alpha
    vignette_clip = ImageClip(vignette).set_duration(duration)

    return CompositeVideoClip([polished, vignette_clip], size=settings.canvas_size)


def _write_video(picture: VideoClip, soundtrack: AudioFileClip, output_path: Path, temp_audiofile: Path) -> None:
    picture = picture.resize(newsize=SETTINGS.canvas_size)
    picture.set_audio(soundtrack).write_videofile(
        str(output_path),
        fps=SETTINGS.target_fps,
        temp_audiofile=str(temp_audiofile),
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="320k",
        bitrate="12M",
        preset="medium",
        ffmpeg_params=[
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-movflags",
            "+faststart",
            "-crf",
            "18",
        ],
        threads=THREADS,
        logger=None,
    )


def generate_video(
    video_paths: List[Path],
    tts_path: Path,
    subtitles_path: Optional[Path] = None,  # kept for signature compatibility
    script: Optional[str] = None,
    search_mappings: Optional[List[dict[str, Any]]] = None,
) -> Path:
    try:
        audio = AudioFileClip(str(tts_path))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to open voiceover at {tts_path}: {exc}") from exc

    duration = audio.duration
    search_mappings = search_mappings or []
    script_segments: List[str] = [
        item.get("line", "").strip()
        for item in search_mappings
        if isinstance(item, dict) and isinstance(item.get("line"), str) and item.get("line", "").strip()
    ]
    if not script_segments and script:
        script_segments = [text for text, _, _ in _approximate_segments_from_script(script, duration)]

    assets = _resolve_assets(video_paths)
    storyboard, storyboard_sources = _build_storyboard(
        assets, duration, SETTINGS, script_segments=script_segments, search_mappings=search_mappings
    )
    layers: List[VideoClip] = []
    layers.extend(_background_layers(duration, SETTINGS))
    layers.extend(storyboard)

    if not (SAFE_MODE or DISABLE_SUBTITLES):
        layers.extend(_build_caption_layers(tts_path, duration, script))

    picture = None
    soundtrack = None
    result = None
    video_id = uuid.uuid4()
    output_path = OUTPUT_PATH / f"{video_id}.mp4"
    temp_audiofile = TEMP_PATH / f"{video_id}.m4a"

    try:
        picture = CompositeVideoClip(layers, size=SETTINGS.canvas_size).set_duration(duration)
        picture = _add_final_polish(picture, duration, SETTINGS)
        soundtrack = create_conversational_soundtrack(tts_path)
        result = picture.set_audio(soundtrack)
        _write_video(result, soundtrack, output_path, temp_audiofile)
    except Exception:
        with suppress(FileNotFoundError):
            output_path.unlink()
        raise
    finally:
        with suppress(FileNotFoundError):
            temp_audiofile.unlink()
        _close_media_resources([*layers, *storyboard_sources, picture, result, soundtrack])
        with suppress(Exception):
            audio.close()
    return output_path


def generate_video_with_karaoke(
    video_paths: List[Path],
    tts_path: Path,
    script: Optional[str] = None,
    search_mappings: Optional[List[dict[str, Any]]] = None,
) -> Path:
    return generate_video(video_paths, tts_path, script=script, search_mappings=search_mappings)


def save_video(video_url: str) -> Path:
    video_id = uuid.uuid4()
    video_path = TEMP_PATH / f"{video_id}.mp4"
    response = requests.get(video_url, timeout=30)
    response.raise_for_status()
    video_path.write_bytes(response.content)
    return video_path


def save_image(image_url: str) -> Path:
    image_id = uuid.uuid4()
    image_path = TEMP_PATH / f"{image_id}.jpg"
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image_path.write_bytes(response.content)
    return image_path
