"""Metadata helpers tuned to match the configured storytelling style."""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List

from config import OUTPUT_PATH, STYLE_METADATA, VIDEO_STYLE
from utils.common import flatten_search_terms

logger = logging.getLogger(__name__)

CINEMATIC_TITLE_FRAGMENTS = [
    "The Incident No One Could Explain -- Until Now",
    "Inside the Case Investigators Still Fear to Name",
    "A Theory That Reshapes What We Know About Consciousness",
    "The Night Something Watched Back",
    "Secrets Archivists Hid in the Dark Wing",
    "When the Cameras Rolled and Reality Bent",
    "The Witness Who Swore the City Changed Color",
    "A File Locked for Forty Years Finally Opens",
]

TITLE_PREFIXES = [
    "The night",
    "When silence",
    "Inside the corridor where",
    "What investigators whispered",
    "The last signal before",
    "An eyewitness to the moment",
]

TITLE_SUFFIXES = [
    "and no one looked away again",
    "still haunts the official report",
    "rewrites the case forever",
    "finally breathes in the light",
    "meets a theory colder than the facts",
    "will never be archived",
]

CONVERSATIONAL_TITLE_FORMS = [
    "The Truth About {title}",
    "What Everyone Gets Wrong About {title}",
    "Breakdown: {title}",
    "Inside {title}",
    "{title} Explained Simply",
    "Why {title} Matters Right Now",
]


def _make_cinematic_title(base_clean: str) -> str:
    if not base_clean:
        return random.choice(CINEMATIC_TITLE_FRAGMENTS)
    if random.random() < 0.5:
        prefix = random.choice(TITLE_PREFIXES)
        suffix = random.choice(TITLE_SUFFIXES)
        return f"{prefix} {base_clean.lower()} {suffix}".strip().title()
    return base_clean


def _make_conversational_title(base_clean: str) -> str:
    if not base_clean:
        template = random.choice(CONVERSATIONAL_TITLE_FORMS)
        return template.format(title="This Big Idea")
    template = random.choice(CONVERSATIONAL_TITLE_FORMS)
    formatted = template.format(title=base_clean.strip().rstrip("."))
    return formatted if formatted else base_clean


def make_viral_title(base: str) -> str:
    """Construct a title that matches the configured VIDEO_STYLE."""
    base_clean = re.sub(r"\s+", " ", base).strip()
    if VIDEO_STYLE == "cinematic":
        return _make_cinematic_title(base_clean)
    return _make_conversational_title(base_clean)


def _safe_filename(text: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text).strip()
    sanitized = re.sub(r"\s+", " ", sanitized)
    if not sanitized:
        return "video"
    return sanitized[:100]


def save_metadata(
    title: str,
    description: str,
    topic: str,
    script: str,
    search_terms: List[Any],
    video_path: Path,
) -> Path:
    """Persist metadata and rename the rendered video in a structured archive."""
    styled_title = make_viral_title(title)
    today = datetime.now().strftime("%Y-%m-%d")

    style_defaults = STYLE_METADATA.get(VIDEO_STYLE, STYLE_METADATA["conversational"])
    flattened_terms = flatten_search_terms(search_terms)
    keywords = list(dict.fromkeys(style_defaults["keywords"] + flattened_terms))[:18]
    keyword_line = ", ".join(keywords)
    tagline_map = {
        "cinematic": "Cinematic mystery",
        "investigative": "Investigative documentary",
    }
    tagline_prefix = tagline_map.get(VIDEO_STYLE, "Smart explainer")
    tagline = f"{tagline_prefix} - {keyword_line}"

    clean_script = script.strip()
    if len(clean_script) > 50000:
        logger.warning("Script too large (%d chars); truncating for metadata.", len(clean_script))
        clean_script = clean_script[:50000] + "\n[truncated]"

    metadata = {
        "title": styled_title,
        "description": description.strip(),
        "topic": topic,
        "script": clean_script,
        "keywords": keywords,
        "tagline": tagline,
        "rendered_at": today,
        "style": VIDEO_STYLE,
    }

    destination_dir = OUTPUT_PATH / today
    destination_dir.mkdir(parents=True, exist_ok=True)
    safe_title = _safe_filename(styled_title)
    new_video_file = destination_dir / f"{safe_title}.mp4"
    shutil.move(str(video_path), new_video_file)

    metadata_path = destination_dir / f"{safe_title}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return new_video_file
