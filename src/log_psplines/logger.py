import sys
import time

from loguru import logger

# Remove loguru's default handler to avoid duplicate logging outputs
logger.remove()

# Configure colors for different log levels (DEBUG will be yellow as requested)
logger.level("DEBUG", color="<d>")
# Make INFO explicitly green (we'll render it without bold in the formatter)
logger.level("INFO", color="<k>")
logger.level("WARNING", color="<yellow>")
logger.level("ERROR", color="<red>")
logger.level("CRITICAL", color="<red>")

# Record module start time so we can print elapsed runtime in logs
_START_TIME = time.time()


# Use a callable formatter so we can inject elapsed mm:ss for each record
def _format(record) -> str:
    # compute elapsed seconds since module import
    elapsed = int(time.time() - _START_TIME)
    minutes = elapsed // 60
    seconds = elapsed % 60
    elapsed_str = f"{minutes:02d}:{seconds:02d}"
    level_name = record["level"].name
    level_part = f"<bold><level>{level_name}</level></bold>"
    message_part = f"<level>{record['message']}</level>"
    # ensure a trailing newline so log entries don't run together
    return f"|{elapsed_str}| <blue>LogPSpline</blue> | {level_part} | {message_part}\n"


# Add a single stdout sink; store handler id for dynamic level changes
_handler_id = logger.add(sys.stdout, format=_format, level="INFO")


def set_level(level: str):
    global _handler_id
    # Remove the current custom handler, then add a new one with requested level
    try:
        logger.remove(_handler_id)
    except Exception:
        # If handler removal fails (e.g., _handler_id invalid), clear all handlers
        logger.remove()
    _handler_id = logger.add(sys.stdout, format=_format, level=level)
