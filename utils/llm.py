"""Language model helpers for engaging, conversational video storytelling."""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from typing import Any, Dict, Iterable, List

import requests
from pydantic import BaseModel, Field, ValidationError
import openai
from openai import OpenAI

from config import (
    GEMINI_API_KEY,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_RETRY_BASE_DELAY,
    LLM_RETRY_MAX_DELAY,
    LLM_THROTTLE_SECONDS,
    NEWS_API_KEY,
    POSSIBLE_TOPICS,
    VIDEO_STYLE,
)

logger = logging.getLogger(__name__)

_DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_CHAT_MODEL_NAME = LLM_MODEL

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url=os.getenv("GEMINI_BASE_URL", _DEFAULT_GEMINI_BASE_URL),
)

_JSON_FORMAT = {"type": "json_object"}
OPENAI_API_EXCEPTION = getattr(openai, "APIError", Exception)
_NEWS_LOCK = threading.Lock()
_LAST_NEWS_FETCH = 0.0
_NEWS_RATE_LIMIT_SECONDS = 60.0

class _ChatRateLimiter:
    """Coordinate chat completion cooldowns across threads."""

    def __init__(self) -> None:
        self._next_allowed = 0.0
        self._quota_reset = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        delay = 0.0
        with self._lock:
            now = time.monotonic()
            delay = max(self._next_allowed - now, self._quota_reset - now, 0.0)
        if delay > 0:
            logger.warning("Delaying LLM call for %.2fs due to rate limits.", delay)
            time.sleep(delay)

    def throttle(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        with self._lock:
            self._next_allowed = max(self._next_allowed, time.monotonic()) + delay_seconds

    def note_quota_reset(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        with self._lock:
            self._quota_reset = max(self._quota_reset, time.monotonic() + delay_seconds)


_STYLE_PROFILES = {
    "conversational": {
        "persona": (
            "You are an engaging video creator who makes informative content that feels like a conversation with a smart friend. "
            "Your style is warm, direct, and energetic while remaining professional. "
            "You address the viewer directly ('you might wonder...'), use simple metaphors, and ground abstract concepts in real examples. "
            "Your content is structured with clear segments, helpful transitions, and delivers clear value with a strong call to action."
        ),
        "engaging_frames": [
            "the question you've probably been asking",
            "what the data actually reveals",
            "the real story behind the headlines",
            "why this matters to you right now",
            "the hidden pattern that changes everything",
        ],
        "topic_keywords": [
            "explained",
            "how it works",
            "the truth about",
            "what you need to know",
            "breakdown",
            "analysis",
            "surprising",
            "why",
            "data",
            "research",
            "real example",
            "step by step",
            "understanding",
            "clear",
            "direct",
        ],
        "script_prompt": (
            'Title: "{title}".\n'
            "Write an engaging, conversational narration that speaks directly to the viewer.\n\n"
            "Requirements:\n"
            "- 500-800 words (~3-5 minutes at a moderate pace).\n"
            "- Start with a compelling hook: ask a direct question to the viewer, then promise insight.\n"
            "- Use direct address: 'you might wonder...', 'you've probably heard...', 'let me show you...'\n"
            "- Structure: 3-4 clear segments, each introduced with transition phrases like 'Next, ...', 'Now let's dive into...', 'Here's where it gets interesting...'\n"
            "- Include 2-3 rhetorical questions to keep the viewer engaged.\n"
            "- Use simple metaphors and analogies to explain complex ideas.\n"
            "- Ground at least one concept in a specific, relatable real-life example.\n"
            "- When mentioning statistics or numbers, call them out clearly for visual emphasis.\n"
            "- Include a brief recap section before the end.\n"
            "- Finish with a clear, actionable call to action.\n"
            "- Pace: moderately fast with natural pauses at key points.\n"
            "- Language: accessible, energetic, but professional. No jargon without explanation.\n"
            "- No section labels, no timestamps, no scene instructions.\n"
            "- Keep sentences varied in length for rhythm."
        ),
        "description_prompt": (
            'Title: "{title}".\n'
            "Script excerpt:\n{excerpt}\n\n"
            "Write a 2-3 sentence YouTube description that clearly explains what viewers will learn. "
            "Tone: helpful, energetic, professional. "
            "Include what problem is solved and what value the viewer gains. Avoid hashtags but include a subtle CTA."
        ),
        "search_prompt": (
            'Title: "{title}".\nScript excerpt:\n{excerpt}\n\n'
            "Break the narration into 12-15 visual beats. "
            "For each beat, provide:\n"
            "- The exact narration line (1 short sentence)\n"
            "- 5-7 highly specific stock search terms that literally match the line's content\n"
            "- 1-2 motion descriptors (e.g., 'slow zoom in', 'fast cut', 'pan left')\n"
            "Blend: lifestyle footage, motion graphics, screen captures, text overlays, macro shots.\n"
            "Focus on concrete nouns and actions from the script. "
            "Example: If script says 'your brain processes data', include terms like 'neurons firing', 'brain scan', 'data flow animation'.\n"
            'Return JSON: {{"search_terms": [{{"line": "...", "terms": ["term1", "term2", "..."], "motion": "..."}}]}}'
        ),
        "tone_references": [
            "YouTube explainers",
            "friendly direct-to-camera educators",
            "data-driven storytelling with direct address",
            "warm conversational productivity channels",
        ],
        "target_duration": "3-5",
    },
    "cinematic": {
        "persona": (
            "You are a cinematic narrator crafting tense, atmospheric mini-documentaries. "
            "Your delivery is deliberate, moody, and investigative while remaining factual. "
            "You lean into sensory description, cinematic pacing, and cliffhanger reveals. "
            "Every beat should feel like peeling back a layer of a mystery."
        ),
        "engaging_frames": [
            "the file investigators buried",
            "the moment the timeline breaks",
            "evidence no one expected",
            "the witness account that chills the room",
            "what the redacted memo really implies",
        ],
        "topic_keywords": [
            "mystery",
            "classified",
            "unexplained",
            "archive",
            "secret",
            "case file",
            "legend",
        ],
        "script_prompt": (
            'Title: "{title}".\n'
            "Write a 350-550 word cinematic narration. "
            "Tone: investigative documentary with suspense.\n"
            "- Start with a cold open that hints at danger or an anomaly.\n"
            "- Use sensory descriptions (sound, light, texture) to set each scene.\n"
            "- Use short, deliberate sentences for tension, then longer reflections for reveals.\n"
            "- Introduce at least two 'evidence' beats (documents, witnesses, anomalies).\n"
            "- Close with an open question or implication rather than a direct CTA.\n"
            "- No dialogue tags, no camera directions, no timestamps.\n"
            "- Avoid sensationalism; stay grounded but moody."
        ),
        "description_prompt": (
            'Title: "{title}".\n'
            "Script excerpt:\n{excerpt}\n\n"
            "Write a 2-3 sentence documentary-style description. "
            "Emphasize the unanswered question, the stakes, and why this mystery matters now. "
            "Tone: cinematic, investigative, no emojis or hashtags."
        ),
        "search_prompt": (
            'Title: "{title}".\nScript excerpt:\n{excerpt}\n\n'
            "Break the narration into 12-15 cinematic beats. "
            "For each beat, include:\n"
            "- a short atmospheric line from the narration\n"
            "- 5-7 specific visual search terms tied to concrete nouns or actions (foggy alley, archive drawer, tungsten spill, macro shadows)\n"
            "- 1-2 motion/shot descriptors (slow dolly, handheld drift, overhead pan, macro pull focus)\n"
            "Prioritize moody environments, rain or fog, timestamp overlays, empty streets, archival textures, and gentle camera moves.\n"
            'Return JSON: {{"search_terms": [{{"line": "...", "terms": ["..."], "motion": "..."}}]}}.'
        ),
        "tone_references": [
            "short-form investigative docs",
            "cinematic TikTok essays",
            "slow-burn true crime explainers",
        ],
        "target_duration": "2-4",
    },
    "investigative": {
        "persona": (
            "You are a long-form investigative storyteller in the style of Arabic true-crime channels "
            "but fully in English. Your narration is calm, deliberate, and incredibly detailed. "
            "You move step by step, never rushing, explaining each detail slowly and logically. "
            "You avoid dramatization; instead, you build tension through facts and careful pacing. "
            "Your tone resembles a documentary narrator who reconstructs events with patience, "
            "often walking the viewer through timelines, decisions, mistakes, and small turning points. "
            "Your voice feels methodical, factual, precise, and emotionally restrained."
        ),
        "engaging_frames": [
            "the clue everyone ignored",
            "the tiny detail that changed everything",
            "a case that took years to understand",
            "a timeline full of small decisions",
            "the moment investigators realized the truth",
        ],
        "topic_keywords": [
            "case",
            "investigation",
            "mystery",
            "crime",
            "solved",
            "unsolved",
            "forensic",
            "timeline",
            "evidence",
            "detail",
        ],
        "script_prompt": (
            'Title: "{title}".\n'
            "Write a 650–900 word investigative narration in English.\n"
            "Tone: patient, factual, slow-paced, extremely detailed — similar to a documentary narrator.\n"
            "Structure:\n"
            "- Begin with a cold, factual introduction summarizing the event or case.\n"
            "- Reconstruct the timeline step by step, focusing on small, mundane details.\n"
            "- Explain what each person did, what they saw, what choices they made, and why.\n"
            "- Include quiet tension built through ordinary facts.\n"
            "- Provide subtle emotional context but no sensationalism.\n"
            "- Describe environment, objects, timestamps, and transitions.\n"
            "- Include at least one moment where new information changes the direction of the case.\n"
            "- End with a calm reflection on what was learned.\n"
            "No camera instructions. No dialogue labels. No dramatization.\n"
            "The pacing should feel slow, methodical, and deeply descriptive.\n"
            "Editorial cues for the video editor: narration must break cleanly into investigative beats that pair with slow pans, timestamp overlays, hands placing objects, quiet hallways, evidence tables, and overhead map/document shots. Insert natural micro-pauses suitable for 2–5% zooms. Avoid dramatization; tension must come from facts and pacing."
        ),
        "description_prompt": (
            'Title: "{title}".\n'
            "Script excerpt:\n{excerpt}\n\n"
            "Write a 2–3 sentence documentary-style YouTube description. "
            "Explain the case being explored and what viewers will understand by the end. "
            "Tone: calm, investigative, factual. No hype, no sensationalism."
        ),
        "search_prompt": (
            'Title: "{title}".\nScript excerpt:\n{excerpt}\n\n'
            "Break the narration into 12-16 investigative beats.\n"
            "For each beat, provide:\n"
            "- one concise narration line capturing the moment\n"
            "- 5-7 investigative-style search terms grounded in objects and actions (timestamp closeup, evidence table overhead, documents macro, quiet hallway night, hands placing folder)\n"
            "- 1-2 motion descriptors (slow pan right, subtle zoom, locked-off, tripod steady)\n\n"
            "Mandatory visual style: slow, smooth, minimal motion; neutral/desaturated palette with cool shadows and slight warmth on skin tones; documentary b-roll; evidence-like macros; quiet interiors or empty streets; overhead shots of documents/maps/timestamps; subtle camera movement (2–5% zoom, slow pans); no neon, no fast movement, no cinematic flares.\n"
            "Use or combine terms like: evidence table overhead, documents macro, cold office interior, timestamp closeup, quiet street night, hands placing object, forensic detail macro, dim hallway slow pan, neutral palette desk, closeup writing hand, overhead map analysis, subtle camera movement.\n"
            'Return JSON: {{"search_terms": [{{"line": "...", "terms": ["...", "..."], "motion": "..."}}]}}.'
        ),
        "tone_references": [
            "slow-paced true-crime explainers",
            "investigative documentary channels",
            "case file storytelling",
        ],
        "target_duration": "8-12",
    },
    "story": {
        "persona": (
            "You are a short-story narrator. "
            "Your stories are simple, emotional, and easy to follow. "
            "You use plain language, short sentences, and a clear twist or lesson."
        ),
        "engaging_frames": [
            "a small moment that changes everything",
            "a quiet mystery in a normal day",
            "a choice that has a cost",
        ],
        "topic_keywords": [
            "short story",
            "moral",
            "twist",
            "life lesson",
            "fiction",
        ],
        "script_prompt": (
            "Write a short fictional story in clear, simple English.\n"
            "Requirements:\n"
            "- 250 to 450 words.\n"
            "- Very clear beginning, problem, and ending.\n"
            "- One main character, simple setting.\n"
            "- Include a gentle twist near the end.\n"
            "- End with a clear takeaway (not preachy).\n"
            "- No scene labels, no timestamps.\n"
        ),
        "description_prompt": (
            'Story title: "{title}".\n'
            "Story excerpt:\n{excerpt}\n\n"
            "Write a short 2-sentence description in English. "
            "Explain the feeling of the story and hint at the lesson."
        ),
        "search_prompt": (
            'Story title: "{title}".\nStory excerpt:\n{excerpt}\n\n'
            "Break the story into 8-10 visual beats.\n"
            "For each beat:\n"
            "- Provide a short line from the narration.\n"
            "- Provide 3-5 simple visual search terms grounded in the line (quiet places, simple objects, silhouettes, closeups).\n"
            "- Add 1 motion descriptor (gentle pan, slow zoom, steady tripod).\n"
            'Return JSON: {{"search_terms": [{{"line": "...", "terms": ["...", "..."], "motion": "..."}}]}}.'
        ),
        "tone_references": [
            "soft narrated fiction",
            "simple moral stories",
            "low-motion b-roll",
        ],
        "target_duration": "2-4",
    },
}

STYLE_PROFILE = _STYLE_PROFILES.get(VIDEO_STYLE, _STYLE_PROFILES["conversational"])
_CHAT_RATE_LIMITER = _ChatRateLimiter()


class TitleResponse(BaseModel):
    titles: List[str] = Field(min_items=1, max_items=10)


class SearchTermItem(BaseModel):
    line: str | None = None
    terms: List[str] = Field(default_factory=list)
    motion: str | None = None


class SearchTermsResponse(BaseModel):
    search_terms: List[SearchTermItem] = Field(min_items=1)


def _validate_model(model_cls, data):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def _model_dump(instance):
    if hasattr(instance, "model_dump"):
        return instance.model_dump()
    return instance.dict()


def _parse_retry_after(response) -> float | None:
    """Parse Retry-After or google retry info if present."""
    if response is None:
        return None
    try:
        retry_after_header = response.headers.get("Retry-After")
    except Exception:
        retry_after_header = None

    if retry_after_header:
        try:
            return float(retry_after_header)
        except (TypeError, ValueError):
            pass

    try:
        payload = response.json()
    except Exception:
        return None

    # Some providers wrap errors in a list, others in a dict
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict):
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


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if hasattr(openai, "RateLimitError") and isinstance(exc, openai.RateLimitError):
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    return "rate limit" in message or "quota" in message or "resource_exhausted" in message


def _should_retry(exc: Exception) -> bool:
    return isinstance(exc, (requests.RequestException, OPENAI_API_EXCEPTION))


def _calculate_chat_retry_delay(attempt: int, exc: Exception) -> float:
    retry_after = _parse_retry_after(getattr(exc, "response", None))
    if retry_after is None:
        retry_after = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
    jitter = random.uniform(0, 0.35 * LLM_RETRY_BASE_DELAY)
    return min(retry_after + jitter, LLM_RETRY_MAX_DELAY)


def _call_chat_model(
    *,
    system_prompt: str | None,
    user_prompt: str,
    response_format: Dict[str, str] | None = None,
    temperature: float = 0.75,
    max_tokens: int = 2048,
) -> str:
    """Thin wrapper around the chat completion API with error handling."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    last_error: Exception | None = None
    for attempt in range(1, max(1, LLM_MAX_RETRIES) + 1):
        _CHAT_RATE_LIMITER.wait()
        try:
            response = client.chat.completions.create(
                model=_CHAT_MODEL_NAME,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Received empty response from language model.")
            if LLM_THROTTLE_SECONDS > 0:
                _CHAT_RATE_LIMITER.throttle(LLM_THROTTLE_SECONDS)
            return content
        except Exception as exc:
            last_error = exc
            if not _should_retry(exc) or attempt >= max(1, LLM_MAX_RETRIES):
                if _is_rate_limit_error(exc):
                    raise RuntimeError(
                        "Language model quota/rate limit reached. Please slow requests or update billing."
                    ) from exc
                raise
            delay = _calculate_chat_retry_delay(attempt, exc)
            if _is_rate_limit_error(exc):
                _CHAT_RATE_LIMITER.note_quota_reset(delay)
            else:
                _CHAT_RATE_LIMITER.throttle(delay)
            logger.warning(
                "LLM request failed (attempt %d/%d): %s. Retrying in %.2fs",
                attempt,
                max(1, LLM_MAX_RETRIES),
                exc,
                delay,
            )
            time.sleep(delay)

    if last_error:
        raise last_error
    raise RuntimeError("Language model request failed without an error message.")


def _parse_json_response(raw: str, *, context: str) -> Dict[str, Any]:
    """Parse a JSON string from the model."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON for %s: %s", context, raw)
        raise ValueError(f"Model response for {context} was not valid JSON.") from exc


def _sanitize_string_list(values: Iterable[Any], *, context: str) -> List[str]:
    """Normalize any iterable into a list of trimmed strings."""
    cleaned = [str(item).strip() for item in values if str(item).strip()]
    if not cleaned:
        raise ValueError(f"Expected at least one string in {context}.")
    return cleaned


# ---------------------------------------------------------------------------
# Topic + title exploration
# ---------------------------------------------------------------------------

def get_topic() -> str:
    """Pick a random engaging topic."""
    if not POSSIBLE_TOPICS:
        raise ValueError("No topics configured in POSSIBLE_TOPICS.")

    engaging_topics = [
        topic
        for topic in POSSIBLE_TOPICS
        if any(keyword in topic.lower() for keyword in STYLE_PROFILE["topic_keywords"])
    ]
    if engaging_topics:
        return random.choice(engaging_topics)
    logger.warning("No keyword-matching topics for style '%s'; falling back to defaults.", VIDEO_STYLE)
    return random.choice(POSSIBLE_TOPICS)


def get_titles(topic: str) -> List[str]:
    """Generate engaging, direct headline options."""
    framing = random.choice(STYLE_PROFILE["engaging_frames"])
    prompt = (
        f"""Topic: "{topic}".
Build 6-8 English-language video titles that are clear, engaging, and promise value.
Tone references: YouTube explainers, educational content, direct-to-camera style.

Guidelines:
- Address the viewer directly or promise clear insight.
- Use active, energetic language without clickbait.
- Stay under 80 characters, sentence case.
- No emojis, no numbered lists.
- Each title should feel like the start of a helpful conversation.
Anchor the hook around: {framing}.

Return JSON: {{"titles": ["...", "..."]}}"""
    )
    raw = _call_chat_model(
        system_prompt=STYLE_PROFILE["persona"],
        user_prompt=prompt,
        response_format=_JSON_FORMAT,
        temperature=0.85,
    )
    payload = _parse_json_response(raw, context="title generation")
    try:
        parsed = _validate_model(TitleResponse, payload)
    except ValidationError as exc:
        raise ValueError(f"Model response for titles was invalid: {payload}") from exc
    return _sanitize_string_list(parsed.titles, context="title generation")


def get_trending_topics() -> List[str]:
    """Fetch trending topics and mix with curated prompts."""
    curated = [
        "why productivity hacks actually fail",
        "the real cost of daily habits",
        "what successful people do differently",
        "how decision fatigue ruins your day",
        "the science behind motivation",
        "why your brain resists change",
    ]
    trending_sources = list(curated)

    if not NEWS_API_KEY:
        return trending_sources

    try:
        _respect_news_rate_limit()
        response = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={
                "country": "us",
                "pageSize": 40,
                "apiKey": NEWS_API_KEY,
                "category": "general",
            },
            timeout=10,
        )
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            for article in articles:
                title = str(article.get("title") or "").strip()
                if not title:
                    continue
                trending_sources.append(title)
    except requests.RequestException as exc:
        logger.warning("Failed to fetch trending topics: %s", exc)

    return trending_sources


def _respect_news_rate_limit() -> None:
    global _LAST_NEWS_FETCH
    with _NEWS_LOCK:
        now = time.monotonic()
        wait = (_LAST_NEWS_FETCH + _NEWS_RATE_LIMIT_SECONDS) - now
        if wait > 0:
            time.sleep(wait)
        _LAST_NEWS_FETCH = time.monotonic()


def analyze_titles(titles: List[str]) -> List[Dict[str, Any]]:
    """Score titles by clarity and engagement."""
    if not titles:
        return []

    listing = "\n".join(f"{idx + 1}. {title}" for idx, title in enumerate(titles))
    prompt = (
        "You are evaluating video titles for engaging, informative content.\n"
        "Score each title 1-10 on:\n"
        "- Directness (does it clearly state what the video is about?)\n"
        "- Engagement (does it make you want to click and learn more?)\n"
        "- Clarity (is the value proposition obvious?)\n"
        "- Energy (does it feel active and dynamic?).\n"
        "Return JSON: {\"analyzed_titles\": [{\"index\": 1, \"total_score\": 34, ...}]}."
    )
    raw = _call_chat_model(
        system_prompt=STYLE_PROFILE["persona"],
        user_prompt=f"{prompt}\n\n{listing}",
        response_format=_JSON_FORMAT,
        temperature=0.5,
    )
    payload = _parse_json_response(raw, context="title scoring")
    analyzed = payload.get("analyzed_titles", [])
    return analyzed if isinstance(analyzed, list) else []


def get_most_engaging_titles(titles: List[str], n: int = 3) -> List[str]:
    """Return top n titles by engagement scoring."""
    if not titles:
        return []
    analysis = analyze_titles(titles)
    if not analysis:
        return titles[:n]

    scored = []
    for data in analysis:
        idx = int(data.get("index", 1)) - 1
        total = data.get("total_score", 0)
        if 0 <= idx < len(titles):
            scored.append((titles[idx], total))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [title for title, _ in scored[:n]]


def get_best_title(titles: List[str]) -> str:
    """Choose the most engaging title."""
    if not titles:
        raise ValueError("Cannot pick a best title from an empty list.")
    finalists = get_most_engaging_titles(titles, n=3) or titles

    listing = "\n".join(f"{idx + 1}. {title}" for idx, title in enumerate(finalists))
    prompt = (
        "Select the single title that best combines clarity, energy, and direct value to the viewer. "
        "Lean toward active, helpful language. "
        "Return JSON: {\"best_title_index\": 2}."
    )
    raw = _call_chat_model(
        system_prompt=STYLE_PROFILE["persona"],
        user_prompt=f"{prompt}\n\n{listing}",
        response_format=_JSON_FORMAT,
        temperature=0.5,
    )
    payload = _parse_json_response(raw, context="best title selection")
    index = payload.get("best_title_index", 1)
    try:
        zero_based = int(index) - 1
        if 0 <= zero_based < len(finalists):
            return finalists[zero_based]
    except (TypeError, ValueError):
        pass
    return finalists[0]


# ---------------------------------------------------------------------------
# Long-form content assets
# ---------------------------------------------------------------------------

def get_script(title: str) -> str:
    """Generate a script aligned with the configured storytelling style."""
    prompt = STYLE_PROFILE["script_prompt"].format(title=title)
    return _call_chat_model(
        system_prompt=STYLE_PROFILE["persona"],
        user_prompt=prompt,
        temperature=0.85,
        max_tokens=2048,
    ).strip()


def get_description(title: str, script: str) -> str:
    """Generate a description aligned with the configured storytelling style."""
    excerpt = script[:600]
    prompt = STYLE_PROFILE["description_prompt"].format(title=title, excerpt=excerpt)
    return _call_chat_model(
        system_prompt=STYLE_PROFILE["persona"],
        user_prompt=prompt,
        temperature=0.7,
        max_tokens=512,
    ).strip()


def get_search_terms(title: str, script: str) -> List[Dict[str, Any]]:
    """Generate dynamic visual prompts for motion graphics and B-roll."""
    excerpt = script[:1000]
    prompt = STYLE_PROFILE["search_prompt"].format(title=title, excerpt=excerpt)
    raw = _call_chat_model(
        system_prompt=STYLE_PROFILE["persona"],
        user_prompt=prompt,
        response_format=_JSON_FORMAT,
        temperature=0.8,
        max_tokens=2048,
    )
    payload = _parse_json_response(raw, context="search term generation")
    container = {"search_terms": payload} if isinstance(payload, list) else payload
    try:
        parsed = _validate_model(SearchTermsResponse, container)
    except ValidationError as exc:
        logger.error("Invalid search term payload: %s", payload)
        raise ValueError("Model response for search terms was invalid.") from exc
    return [_model_dump(item) for item in parsed.search_terms]


def get_search_terms_flat(title: str, script: str) -> List[str]:
    """Legacy helper kept for compatibility (flattened search terms)."""
    structured = get_search_terms(title, script)
    flattened: List[str] = []
    for item in structured:
        if isinstance(item, dict):
            line = item.get("line")
            if isinstance(line, str):
                flattened.append(line.strip())
            for term in item.get("terms", []):
                if isinstance(term, str):
                    flattened.append(term.strip())
    return [term for term in flattened if term]


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

def generate_viral_video_assets() -> Dict[str, Any]:
    """Complete conversational pipeline for downstream modules."""
    trending = get_trending_topics()
    topic = random.choice(trending) if trending else get_topic()
    titles = get_titles(topic)
    if not titles:
        raise ValueError("No titles generated.")
    best_title = get_best_title(titles)
    script = get_script(best_title)
    description = get_description(best_title, script)
    search_terms = get_search_terms(best_title, script)

    color_grade = {
        "contrast": {"lift_blacks": -5, "soft_highlights": 5},
        "temperature_shift_kelvin": 200,
        "saturation": {"global_pct": 5, "blues_pct": -10},
        "curves": "gentle_s_curve",
        "sharpen_pct": 8,
        "vignette": {"amount": -0.2, "feather": 0.7},
        "grain_pct": 3,
    }

    return {
        "topic": topic,
        "title": best_title,
        "script": script,
        "description": description,
        "search_terms": search_terms,
        "style": VIDEO_STYLE,
        "target_duration_minutes": STYLE_PROFILE.get("target_duration", "3-5"),
        "tone_reference": STYLE_PROFILE.get("tone_references", []),
        "color_grade": color_grade,
    }
