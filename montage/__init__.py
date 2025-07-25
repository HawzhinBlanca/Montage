# Real AI-powered video processing pipeline

# Configure secure logging on import
try:
    from .utils.logging_config import configure_secure_logging
    from .settings import get_settings
    
    settings = get_settings()
    configure_secure_logging(
        level=settings.logging.level,
        log_file=settings.logging.file_path,
        use_json=settings.logging.use_json_format,
        mask_secrets=settings.logging.mask_secrets
    )
except ImportError:
    # Fallback to basic logging if secure logging not available
    import logging
    logging.basicConfig(level=logging.INFO)
