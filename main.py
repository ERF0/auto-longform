import logging
import shutil
import time

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from tqdm import tqdm

from config import (
    CRON_SCHEDULE,
    RUN_ONCE,
    TEMP_PATH,
    TEST_MODE,
    VIDEO_COUNT,
    ensure_directories,
    setup,
)
from utils.audio import generate_voiceover
from utils.llm import (
    get_description,
    get_most_engaging_titles,
    get_script,
    get_search_terms,
    get_titles,
    get_topic,
)
from utils.metadata import save_metadata
from utils.notifications import send_error_notification, send_success_notification
from utils.stock_videos import get_stock_videos
from utils.video import generate_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _log_preview(label: str, text: str, unit: str = "chars") -> None:
    if not text:
        logger.info("%s is empty", label)
        return
    snippet = text[:240] + ("…" if len(text) > 240 else "")
    tokens = len(text.split())
    metric = len(text) if unit == "chars" else tokens
    logger.info("%s (%d %s)", label, metric, unit)
    logger.debug("%s preview: %s", label, snippet)


def _blend_visual_assets(video_assets):
    return video_assets


def generate_video_data(title):
    logger.info("[Generated Title] %s", title)

    script = get_script(title)
    _log_preview("[Generated Script]", script, unit="words")

    description = get_description(title, script)
    _log_preview("[Generated Description]", description)

    search_terms = get_search_terms(title, script)
    logger.info("[Generated Search Terms] %d beats", len(search_terms))
    logger.debug(search_terms)

    stock_videos = get_stock_videos(search_terms)
    logger.info("[Downloaded %d stock videos]", len(stock_videos))

    visual_assets = _blend_visual_assets(stock_videos)
    logger.info("[Prepared %d visual assets]", len(visual_assets))

    voiceover = generate_voiceover(script)
    logger.info("[Generated Voiceover]")

    return title, description, script, search_terms, visual_assets, voiceover


def generate_videos(n: int = 4, metrics: dict | None = None) -> dict:
    metrics = metrics or {}
    metrics.update({"requested_videos": n, "start_time": time.time()})
    try:
        topic = get_topic()

        logger.info("[Generated Topic] %s", topic)

        possible_titles = get_titles(topic)
        logger.info("[Generated Possible Titles] %d options", len(possible_titles))
        logger.debug(possible_titles)

        titles = get_most_engaging_titles(possible_titles, n)

        videos_generated = 0
        metrics["success_count"] = 0
        failed_titles: list[tuple[str, str]] = []
        for title in tqdm(titles, desc="Generating videos"):
            try:
                (
                    title,
                    description,
                    script,
                    search_terms,
                    visual_assets,
                    voiceover,
                ) = generate_video_data(title)

                video = generate_video(visual_assets, voiceover, script=script, search_mappings=search_terms)
                logger.info("[Generated Video]")

                new_video_file = save_metadata(
                    title, description, topic, script, search_terms, video
                )
                logger.info("[Saved Video] %s", new_video_file)
                videos_generated += 1

            except Exception as e:
                error_msg = f"Failed to generate video '{title}'"
                logger.error(error_msg, exc_info=True)
                failed_titles.append((title, str(e)))

        metrics["success_count"] = videos_generated
        metrics["end_time"] = time.time()
        if videos_generated > 0:
            success_msg = (
                f"Successfully generated {videos_generated} video(s)"
            )
            logger.info(success_msg)
            send_success_notification(success_msg, "Video Generation")
        else:
            error_msg = "No videos were successfully generated"
            logger.error(error_msg)
            send_error_notification(error_msg, context="Video Generation")
        if failed_titles:
            preview = "; ".join(f"{title}: {reason}" for title, reason in failed_titles[:5])
            send_error_notification(
                f"{len(failed_titles)} video(s) failed: {preview}",
                context="Video Generation",
            )

    except Exception as e:
        error_msg = "Failed to start video generation process"
        logger.error(f"{error_msg}: {e}")
        send_error_notification(error_msg, e, "Video Generation")
    finally:
        shutil.rmtree(TEMP_PATH, ignore_errors=True)
        ensure_directories([TEMP_PATH])
    return metrics


def main():
    setup()
    if TEST_MODE:
        logger.info("TEST_MODE enabled. External integrations should be stubbed.")

    cron_schedule = CRON_SCHEDULE
    run_once = RUN_ONCE
    video_count = VIDEO_COUNT

    if run_once:
        logger.info("RUN_ONCE is enabled, generating videos immediately...")
        generate_videos(video_count)
        logger.info("Video generation completed. Exiting.")
        return

    logger.info(f"Starting scheduler with cron schedule: {cron_schedule}")
    scheduler = BlockingScheduler()

    trigger = CronTrigger.from_crontab(cron_schedule)
    scheduler.add_job(
        func=generate_videos, trigger=trigger, args=[video_count], id="video_generation"
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        scheduler.shutdown()
    except Exception as e:
        error_msg = "Scheduler failed unexpectedly"
        logger.error(f"{error_msg}: {e}")
        send_error_notification(error_msg, e, "Scheduler")


if __name__ == "__main__":
    main()
