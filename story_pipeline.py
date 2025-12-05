"""One-call helper for generating narrated short-story videos."""

from __future__ import annotations

from pathlib import Path
from typing import List

from config import setup
from utils.audio import generate_voiceover
from utils.llm import get_description, get_script, get_search_terms
from utils.metadata import save_metadata
from utils.stock_videos import get_stock_videos
from utils.video import generate_video


def _blend_assets(video_assets: List[Path], image_assets: List[Path]) -> List[Path]:
    return video_assets or image_assets


def make_story_video(title: str, topic: str = "short story") -> Path:
    """Generate a narrated story video end-to-end."""
    setup()
    script = get_script(title)
    description = get_description(title, script)
    search_terms = get_search_terms(title, script)

    video_assets = get_stock_videos(search_terms)
    visual_assets = _blend_assets(video_assets, [])

    voiceover_path = generate_voiceover(script)
    video_path = generate_video(visual_assets, voiceover_path, script=script, search_mappings=search_terms)

    final_path = save_metadata(
        title=title,
        description=description,
        topic=topic,
        script=script,
        search_terms=search_terms,
        video_path=video_path,
    )
    return final_path
