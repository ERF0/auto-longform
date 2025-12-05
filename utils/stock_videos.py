"""Stock asset helpers that align with the configured storytelling style."""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine, Iterable, List, Sequence, TypeVar

import aiohttp
from moviepy.editor import VideoFileClip

from config import PEXELS_API_KEY, STOCK_FOOTAGE_CONFIG, TEMP_PATH, VIDEO_STYLE
from utils.common import flatten_search_terms

logger = logging.getLogger(__name__)

T = TypeVar("T")

PEXELS_VIDEO_ENDPOINT = "https://api.pexels.com/videos/search"
PEXELS_IMAGE_ENDPOINT = "https://api.pexels.com/v1/search"

VIDEO_TARGET_ASSET_COUNT = 12
IMAGE_TARGET_ASSET_COUNT = 12
MAX_NORMALIZED_TERMS = 35
MAX_PEXELS_FAILURES = 24

STYLE_VIDEO_MODIFIERS = {
    "conversational": [
        "bright lighting",
        "dynamic b-roll",
        "clean background",
        "tech macro",
        "city commute",
        "workspace overhead",
        "hands typing",
        "coffee shop",
        "nature timelapse",
        "data visualization",
    ],
    "cinematic": [
        "slow movement",
        "fog",
        "night ambience",
        "macro shadows",
        "gentle pan",
        "rain window",
        "empty street",
        "archive footage",
        "flickering light",
        "misty forest",
    ],
    "investigative": [
        "forensic detail",
        "slow pan",
        "timestamp overlay",
        "evidence table",
        "overhead desk",
        "documents spread",
        "magnifying glass",
        "microscope view",
        "case file",
        "dark hallway",
    ],
}

STYLE_IMAGE_MODIFIERS = {
    "conversational": [
        "minimal design",
        "motion graphics",
        "white background",
        "kinetic typography",
    ],
    "cinematic": [
        "grainy texture",
        "low key lighting",
        "desaturated palette",
        "moody portrait",
    ],
    "investigative": [
        "neutral palette",
        "documentary still",
        "evidence closeup",
        "map desk overhead",
    ],
}

STYLE_FALLBACK_TERMS = {
    "conversational": ["modern workspace", "city skyline", "hands using tech", "data visualization"],
    "cinematic": ["foggy city street", "mystery archive", "slow motion rain", "hazy hallway"],
    "investigative": [
        "evidence table",
        "empty street at night",
        "quiet hallway",
        "documents on desk",
        "crime lab closeup",
        "timestamp overlay",
    ],
}

STYLE_REQUIRED_TERMS = {
    "investigative": [
        "evidence table overhead",
        "documents macro",
        "timestamp closeup",
        "cold office interior",
        "dim hallway slow pan",
        "subtle camera movement",
    ]
}

VIDEO_TARGET_ASSET_COUNT = int(STOCK_FOOTAGE_CONFIG.get("target_video_clips", VIDEO_TARGET_ASSET_COUNT))
IMAGE_TARGET_ASSET_COUNT = int(STOCK_FOOTAGE_CONFIG.get("target_image_clips", IMAGE_TARGET_ASSET_COUNT))


def _run_async(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    try:
        return asyncio.run(factory())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(factory())
        finally:
            loop.close()


def _normalize_terms(search_terms: Iterable[Any]) -> List[str]:
    prioritized: List[str] = []
    seen: set[str] = set()
    for term in flatten_search_terms(search_terms):
        term_clean = term.strip()
        if not term_clean:
            continue
        if len(term_clean) > 48 or term_clean.count(" ") > 7:
            continue
        term_lower = term_clean.lower()
        if term_lower in seen:
            continue
        seen.add(term_lower)
        prioritized.append(term_clean)
    required = STYLE_REQUIRED_TERMS.get(VIDEO_STYLE, [])
    for term in required:
        lower = term.lower()
        if lower not in seen:
            prioritized.append(term)
            seen.add(lower)
    if prioritized:
        return prioritized[:MAX_NORMALIZED_TERMS]
    return STYLE_FALLBACK_TERMS.get(VIDEO_STYLE, STYLE_FALLBACK_TERMS["conversational"])


def _build_queries(term: str, modifiers: Sequence[str]) -> List[str]:
    queries = [term]
    for modifier in modifiers[:3]:
        queries.append(f"{term} {modifier}")
    return queries


async def _fetch_json(session: aiohttp.ClientSession, url: str, **params: Any) -> Any:
    async with session.get(url, params=params) as response:
        response.raise_for_status()
        return await response.json()


def _estimate_luminance(hex_color: str) -> float:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return 128.0
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _video_color_filter(luminance: float) -> bool:
    if VIDEO_STYLE == "cinematic":
        return luminance <= 160
    if VIDEO_STYLE == "investigative":
        return True  # allow broader palette for evidence-style clips
    return luminance >= 80


def _image_color_filter(luminance: float) -> bool:
    if VIDEO_STYLE == "cinematic":
        return luminance <= 150
    return luminance >= 90


async def _fetch_video_urls(session: aiohttp.ClientSession, query: str, per_page: int = 5) -> List[str]:
    params = {
        "query": query,
        "orientation": "landscape",
        "per_page": per_page,
        "min_duration": 3,
        "max_duration": 12,
        "size": "large",
    }
    payload = await _fetch_json(session, PEXELS_VIDEO_ENDPOINT, **params)
    videos = payload.get("videos", [])
    urls: List[str] = []
    for video in videos:
        color = video.get("color") or "#1a1a1a"
        if not _video_color_filter(_estimate_luminance(color)):
            continue
        files = video.get("video_files", [])
        files = [
            item
            for item in files
            if str(item.get("link", "")).endswith(".mp4")
            and int(item.get("width", 0)) >= int(item.get("height", 0))
            and int(item.get("width", 0)) > 0
        ]
        files.sort(key=lambda item: (int(item.get("height", 0)), int(item.get("width", 0))), reverse=True)
        if not files:
            continue
        urls.append(files[0]["link"])
    return urls


async def _fetch_with_fallback(
    session: aiohttp.ClientSession,
    query: str,
    fallback_terms: List[str],
) -> List[str]:
    """Try primary query, then fall back to broader terms if needed."""
    try:
        urls = await _fetch_video_urls(session, query)
        if len(urls) >= 3:
            return urls
    except Exception as exc:
        logger.debug("Primary query '%s' failed: %s", query, exc)

    modifiers = STYLE_VIDEO_MODIFIERS.get(VIDEO_STYLE, [])
    for term in fallback_terms[:2]:
        modifier = random.choice(modifiers) if modifiers else ""
        broad_query = f"{term} {modifier}".strip()
        try:
            urls = await _fetch_video_urls(session, broad_query)
            if urls:
                logger.info("Fallback query '%s' succeeded", broad_query)
                return urls
        except Exception as exc:
            logger.debug("Fallback query '%s' failed: %s", broad_query, exc)
            continue

    return []


async def _fetch_image_urls(session: aiohttp.ClientSession, query: str, per_page: int = 3) -> List[str]:
    params = {"query": query, "orientation": "portrait", "per_page": per_page}
    payload = await _fetch_json(session, PEXELS_IMAGE_ENDPOINT, **params)
    photos = payload.get("photos", [])
    urls: List[str] = []
    for photo in photos:
        avg_color = photo.get("avg_color") or "#999999"
        if not _image_color_filter(_estimate_luminance(avg_color)):
            continue
        sources = photo.get("src") or {}
        for key in ("portrait", "large2x", "original", "large"):
            candidate = sources.get(key)
            if candidate:
                urls.append(candidate)
                break
    return urls


async def _download_file(
    session: aiohttp.ClientSession,
    url: str,
    suffix: str,
) -> Path:
    asset_id = uuid.uuid4()
    destination = TEMP_PATH / f"{asset_id}{suffix}"
    async with session.get(url) as response:
        response.raise_for_status()
        header = b""
        with destination.open("wb") as f:
            async for chunk in response.content.iter_chunked(64 * 1024):
                if not chunk:
                    continue
                if len(header) < 16:
                    header += chunk[:16 - len(header)]
                f.write(chunk)

        if suffix == ".mp4":
            if not header or len(header) < 12:
                destination.unlink(missing_ok=True)
                raise ValueError("Invalid MP4 file (empty or missing header)")
            if header[4:8] != b"ftyp":
                logger.warning("Downloaded file does not appear to be a valid MP4: %s", url)
                destination.unlink(missing_ok=True)
                raise ValueError("Invalid MP4 file")

    if suffix == ".mp4":
        test_clip = None
        try:
            test_clip = VideoFileClip(str(destination))
        except Exception as exc:
            logger.warning("Downloaded video file is corrupted: %s", exc)
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
            raise
        finally:
            if test_clip is not None:
                try:
                    test_clip.close()
                except Exception:
                    pass
    return destination


async def _download_batch(
    session: aiohttp.ClientSession,
    urls: Sequence[str],
    suffix: str,
    remaining: int,
) -> List[Path]:
    if remaining <= 0:
        return []
    tasks = [
        asyncio.create_task(_download_file(session, url, suffix))
        for url in urls[:remaining]
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assets: List[Path] = []
    for result in results:
        if isinstance(result, Path):
            assets.append(result)
        else:
            logger.warning("Asset download failed: %s", result)
    return assets


async def _collect_assets(
    terms: List[str],
    modifiers: Sequence[str],
    extractor: Callable[[aiohttp.ClientSession, str], Coroutine[Any, Any, List[str]]],
    suffix: str,
    target_count: int,
    fallback_terms: Sequence[str] | None = None,
) -> List[Path]:
    headers = {"Authorization": PEXELS_API_KEY}
    timeout = aiohttp.ClientTimeout(total=45)
    connector = aiohttp.TCPConnector(limit=8)
    assets: List[Path] = []

    failure_count = 0
    async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
        for term in terms:
            if failure_count >= MAX_PEXELS_FAILURES:
                logger.warning("Stopping remote stock lookup after %d failures.", failure_count)
                break
            queries = _build_queries(term, modifiers)
            for query in queries:
                if failure_count >= MAX_PEXELS_FAILURES:
                    break
                try:
                    if fallback_terms is not None and extractor is _fetch_video_urls:
                        urls = await _fetch_with_fallback(session, query, list(fallback_terms))
                    else:
                        urls = await extractor(session, query)
                except aiohttp.ClientResponseError as exc:
                    logger.warning("Pexels API error for '%s': %s", query, exc)
                    failure_count += 1
                    continue
                except Exception as exc:
                    logger.warning("Unexpected error during stock lookup for '%s': %s", query, exc)
                    failure_count += 1
                    continue
                if not urls:
                    continue
                downloads = await _download_batch(
                    session, urls, suffix, target_count - len(assets)
                )
                assets.extend(downloads)
                if len(assets) >= target_count:
                    return assets
    return assets


def get_stock_videos(search_terms: List[dict[str, Any]] | List[str]) -> List[Path]:
    """Download portrait videos matching the configured VIDEO_STYLE."""
    terms = _normalize_terms(search_terms)
    modifiers = STYLE_VIDEO_MODIFIERS.get(VIDEO_STYLE, STYLE_VIDEO_MODIFIERS["conversational"])

    def _runner() -> Coroutine[Any, Any, List[Path]]:
        return _collect_assets(
            terms,
            modifiers,
            _fetch_video_urls,
            ".mp4",
            VIDEO_TARGET_ASSET_COUNT,
            STYLE_FALLBACK_TERMS.get(VIDEO_STYLE, []),
        )

    try:
        assets = _run_async(_runner)
    except Exception as exc:
        logger.warning("Stock video lookup failed: %s", exc)
        assets = []

    if assets:
        logger.info("Downloaded %d stock video asset(s).", len(assets))
    else:
        logger.warning("No stock videos retrieved; falling back to local/secondary footage.")
    return assets


def get_stock_images(search_terms: List[dict[str, Any]] | List[str]) -> List[Path]:
    """Download still images that can be used as B-roll overlays."""
    terms = _normalize_terms(search_terms)
    modifiers = STYLE_IMAGE_MODIFIERS.get(VIDEO_STYLE, STYLE_IMAGE_MODIFIERS["conversational"])

    def _runner() -> Coroutine[Any, Any, List[Path]]:
        return _collect_assets(terms, modifiers, _fetch_image_urls, ".jpg", IMAGE_TARGET_ASSET_COUNT)

    try:
        assets = _run_async(_runner)
    except Exception as exc:
        logger.warning("Stock image lookup failed: %s", exc)
        assets = []

    if assets:
        logger.info("Downloaded %d stock image asset(s).", len(assets))
    else:
        logger.warning("No stock images retrieved; proceeding without remote stills.")
    return assets
