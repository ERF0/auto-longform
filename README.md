
AUTO LONG-FORM VIDEO GENERATION
AI-driven cinematic video pipeline

Overview
This repository is an end-to-end automated video generator:
- Generates topics, titles, scripts, descriptions, and visual beats via LLMs.
- Pulls and filters stock footage from Pexels based on semantic search terms and style.
- Synthesizes voiceovers using Gemini TTS with robust retry and normalization.
- Renders cinematic videos with motion, overlays, safe fallbacks, and caption support.
- Packages everything with titles, metadata, and structured archives.

Features
- Content Engine: Topic, title, script, description, and search-term generation via AI.
- Voiceover: Gemini TTS to WAV with normalization and retry logic.
- Stock Footage: Style-aware search on Pexels with fallback.
- Video Renderer: Motion, zooms, gradients, grain, overlays, and captioning.
- Metadata Layer: Viral-style titles, tags, keyword extraction.
- Orchestration: Cron scheduling, batch mode, notifications.

Architecture
1. Content Engine (LLM)
2. Stock Footage Engine
3. Voiceover System
4. Video Renderer
5. Metadata & Archiving
6. Orchestration via main.py

Storytelling Styles
- conversational
- cinematic
- investigative
- story

Getting Started
Install dependencies:
    pip install -r requirements.txt

Configure environment:
    export GEMINI_API_KEY=...
    export PEXELS_API_KEY=...
    export VIDEO_STYLE=conversational

Run generator:
    RUN_ONCE=true VIDEO_COUNT=1 python main.py

Output Structure:
output/YYYY-MM-DD/
    video.mp4
    metadata.json
    script.txt (optional)

License
MIT License
