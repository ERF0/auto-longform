import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import dotenv

dotenv.load_dotenv()


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() == "true"


VIDEO_STYLE = os.environ.get("VIDEO_STYLE", "conversational").strip().lower()
_VALID_VIDEO_STYLES = {"conversational", "cinematic", "story", "investigative"}
if VIDEO_STYLE not in _VALID_VIDEO_STYLES:
    VIDEO_STYLE = "conversational"

STYLE_TOPIC_PRESETS: Dict[str, List[str]] = {
    "conversational": [
        "Habits That Actually Compound Results",
        "Psychology Tricks For Negotiations",
        "Space Milestones Explained Simply",
        "Hidden Patterns Behind Viral Innovations",
        "Why Certain Productivity Systems Fail",
        "Technology Shifts Changing Workflows",
        "Science Facts You Can Use Today",
        "Money Myths That Refuse To Die",
        "Decisions That Shape Elite Performers",
        "Data Stories The News Ignored",
    ],
    "cinematic": [
        "Unsolved Signals From Declassified Missions",
        "Vanished Expeditions In Ice Fields",
        "Psychology Experiments That Went Missing",
        "Rogue Inventors And Their Secret Labs",
        "Artifacts Whispered About In Archives",
        "Cities That Reported Impossible Lights",
        "Cold Cases Reopened By Algorithms",
        "Witness Reports That Bent Reality",
        "Locked Rooms Found Years Later",
        "Numbers That Should Not Exist",
    ],
    "story": [
        "A short story about hope in a quiet town",
        "A simple moral about keeping a promise",
        "A small mystery during an ordinary day",
        "A choice that changes a friendship",
        "A family memory that returns years later",
        "A walk home that reveals a secret",
        "A letter that arrives too late",
        "A stranger who leaves a lesson behind",
        "A moment of kindness with a hidden cost",
        "A quiet decision that shapes a life",
    ],
    "investigative": [
        "A cold case reopened after a forgotten clue",
        "How a small mistake unraveled an alibi",
        "The routine call that changed an investigation",
        "A missing file that rewrote a timeline",
        "An overlooked witness detail years later",
        "Tracing evidence through mundane errands",
        "The paperwork error that solved a mystery",
        "How timestamps exposed the real sequence",
        "Reconstructing a crime from minor objects",
        "What investigators missed the first time",
    ],
}

POSSIBLE_TOPICS = STYLE_TOPIC_PRESETS[VIDEO_STYLE]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY_AUTO_YT_SHORTS")
if OPENAI_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

LLM_MODEL = (
    os.environ.get("LLM_MODEL")
    or os.environ.get("GEMINI_CHAT_MODEL")
    or os.environ.get("OPENAI_MODEL")
    or "gemini-2.0-flash"
)

LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "4"))
LLM_RETRY_BASE_DELAY = float(os.environ.get("LLM_RETRY_BASE_DELAY", "2.0"))
LLM_RETRY_MAX_DELAY = float(os.environ.get("LLM_RETRY_MAX_DELAY", "20.0"))
LLM_THROTTLE_SECONDS = float(os.environ.get("LLM_THROTTLE_SECONDS", "0.0"))

GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
GEMINI_TTS_MAX_RETRIES = int(os.environ.get("GEMINI_TTS_MAX_RETRIES", "4"))
GEMINI_TTS_RETRY_BASE_DELAY = float(os.environ.get("GEMINI_TTS_RETRY_BASE_DELAY", "2.0"))
GEMINI_TTS_RETRY_MAX_DELAY = float(os.environ.get("GEMINI_TTS_RETRY_MAX_DELAY", "30.0"))
GEMINI_TTS_THROTTLE_SECONDS = float(os.environ.get("GEMINI_TTS_THROTTLE_SECONDS", "0.0"))

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")

ASSEMBLY_AI_API_KEY = os.environ.get("ASSEMBLY_AI_API_KEY")

NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

TEMP_PATH = Path(os.environ.get("TEMP_PATH", "temp"))

OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "output"))

BACKGROUND_SONGS_PATH = Path("music")

SECONDARY_CONTENT_PATH = Path(os.environ.get("SECONDARY_CONTENT_PATH", "secondary_video"))

CRON_SCHEDULE = os.environ.get("CRON_SCHEDULE", "31 4 * * *")

RUN_ONCE = _env_flag("RUN_ONCE", "false")

VIDEO_COUNT = int(os.environ.get("VIDEO_COUNT", "1"))

APPRISE_URL = os.environ.get("APPRISE_URL")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

NO_UPLOAD = _env_flag("NO_UPLOAD", "false")
NOTIFY_ON_SUCCESS = _env_flag("NOTIFY_ON_SUCCESS", "false")

# Memory/quality toggles
LOW_MEMORY_MODE = _env_flag("LOW_MEMORY_MODE", "false")

# Safe mode toggles for downstream video rendering
SAFE_MODE = _env_flag("SAFE_MODE", "false")
DISABLE_SUBTITLES = _env_flag("DISABLE_SUBTITLES", "false")

# Video/caption appearance knobs
# Landscape for longform (width, height)
if LOW_MEMORY_MODE:
    CANVAS_SIZE: Tuple[int, int] = (960, 540)
    TARGET_FPS = 24
    THREADS = 1
else:
    CANVAS_SIZE: Tuple[int, int] = (1920, 1080)
    TARGET_FPS = int(os.environ.get("TARGET_FPS", "30"))
    THREADS = min(8, os.cpu_count() or 1)

SEGMENT_DURATION_RANGE: Tuple[float, float] = (3.5, 6.5)
VISUAL_SEGMENT_RANGE: Tuple[float, float] = (3.0, 7.0)
TRANSITION_DURATION = float(os.environ.get("TRANSITION_DURATION", "0.9"))

CAPTION_FONT_PATH = os.environ.get("CAPTION_FONT_PATH", "fonts/serif_caption.ttf")
CAPTION_FONT_SIZE = int(os.environ.get("CAPTION_FONT_SIZE", "72"))
CAPTION_PADDING = int(os.environ.get("CAPTION_PADDING", "34"))
CAPTION_MAX_WIDTH_RATIO = float(os.environ.get("CAPTION_MAX_WIDTH_RATIO", "0.88"))
CAPTION_COLOR = os.environ.get("CAPTION_COLOR", "white")
CAPTION_STROKE = os.environ.get("CAPTION_STROKE", "#111111")
CAPTION_BACKGROUND = os.environ.get("CAPTION_BACKGROUND", "#050505")
CAPTION_BG_OPACITY = float(os.environ.get("CAPTION_BG_OPACITY", "0.24"))
CAPTION_FADE = float(os.environ.get("CAPTION_FADE", "0.65"))

CANVAS_TINT = (8, 12, 18)
HIGHLIGHT_TINT = (22, 28, 38)
FILM_GRAIN_OPACITY = 0.08
LETTERBOX_HEIGHT = 90

STOCK_FOOTAGE_CONFIG = {
    "target_video_clips": 15,
    "target_image_clips": 15,
    "enable_semantic_matching": True,
    "overlay_frequency": 0.6,
    "split_screen_frequency": 0.15,
    "segment_duration_multiplier": 0.75,
}

STYLE_METADATA = {
    "conversational": {
        "keywords": [
            "explainer",
            "education",
            "data story",
            "science",
            "strategy",
            "psychology",
            "tools",
            "productivity",
            "insight",
        ]
    },
    "cinematic": {
        "keywords": [
            "mystery",
            "cinematic",
            "documentary",
            "unexplained",
            "psychology",
            "hidden history",
            "dark science",
        ]
    },
    "investigative": {
        "keywords": [
            "investigative",
            "case file",
            "forensic",
            "timeline",
            "evidence",
            "cold case",
            "true crime",
            "detailed analysis",
            "documentary",
        ]
    },
}

TEST_MODE = _env_flag("TEST_MODE", "false")


def ensure_directories(paths: Iterable[Path] | None = None) -> None:
    """Create writable directories lazily when the application starts."""
    candidates = list(paths) if paths is not None else [TEMP_PATH, OUTPUT_PATH]
    for path in candidates:
        if not path:
            continue
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Unable to create directory {path}: {exc}") from exc


def validate_config() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY required for narration")
    if not TEMP_PATH.exists() or not OUTPUT_PATH.exists():
        raise RuntimeError("Call ensure_directories() before validating config")
    if VIDEO_COUNT <= 0:
        raise ValueError("VIDEO_COUNT must be positive")
    if VIDEO_STYLE not in _VALID_VIDEO_STYLES:
        raise ValueError("VIDEO_STYLE must be conversational or cinematic")

    if not PEXELS_API_KEY:
        logger.warning("PEXELS_API_KEY missing; stock asset downloads disabled")
    if not ASSEMBLY_AI_API_KEY:
        logger.warning("ASSEMBLY_AI_API_KEY missing; using script-derived subtitles")


def setup() -> None:
    """Initialize filesystem state and validate critical configuration."""
    ensure_directories()
    validate_config()
