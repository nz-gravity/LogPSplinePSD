import sys

from loguru import logger

# Remove loguru's default handler to avoid duplicate logging outputs
logger.remove()

# Configure colors for different log levels (DEBUG will be yellow as requested)
logger.level("DEBUG", color="<d>")
logger.level("INFO", color="<k>")
logger.level("WARNING", color="<yellow>")
logger.level("ERROR", color="<red>")
logger.level("CRITICAL", color="<red>")

# Use the <level> tag so both the level text and the message are rendered using the level's color
_log_format = "<blue>LogPSpline</blue> | <bold><level>{level}</level></bold> | <level>{message}</level>"
# Add a single stdout sink; store handler id for dynamic level changes
_handler_id = logger.add(sys.stdout, format=_log_format, level="INFO")


def set_level(level: str):
    global _handler_id
    # Remove the current custom handler, then add a new one with requested level
    try:
        logger.remove(_handler_id)
    except Exception:
        # If handler removal fails (e.g., _handler_id invalid), clear all handlers
        logger.remove()
    _handler_id = logger.add(sys.stdout, format=_log_format, level=level)
