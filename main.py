from core.settings import SettingsError, load_settings
from observability.logger import get_logger


def main() -> int:
    logger = get_logger(__name__)
    try:
        settings = load_settings()
    except SettingsError as error:
        logger.error("Configuration error: %s", error)
        return 1

    logger.info("Loaded settings for %s", settings.app.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
