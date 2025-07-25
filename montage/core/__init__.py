# Core business logic module
import logging
import os

try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration

    # Configure Sentry with logging integration
    sentry_logging = LoggingIntegration(
        level=logging.INFO,  # Capture info and above as breadcrumbs
        event_level=logging.ERROR,  # Send errors as events
    )

    # Initialize Sentry if DSN is provided
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[sentry_logging],
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            environment=os.getenv("ENVIRONMENT", "development"),
        )
        logging.info("Sentry initialized successfully")
except ImportError:
    logging.warning("Sentry SDK not available - error tracking disabled")

# Configure structured logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO"),
    handlers=[
        logging.StreamHandler(),
        # Add file handler if LOG_FILE is set
        *(
            [logging.FileHandler(os.getenv("LOG_FILE"))]
            if os.getenv("LOG_FILE")
            else []
        ),
    ],
)
