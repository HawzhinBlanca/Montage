#!/usr/bin/env python3
"""
API service wrappers with comprehensive rate limiting integration.

This module provides rate-limited wrappers for all external API services
to prevent cost explosions and ensure production reliability.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

from .rate_limiter import (
    rate_limited,
    Priority,
    RateLimitConfig,
    get_rate_limiter,
    rate_limit_manager,
)
from .cost import priced, BudgetExceededError
from .rate_limit_config import get_service_config, env_config

logger = logging.getLogger(__name__)


class ServiceWrapper:
    """Base class for API service wrappers with rate limiting"""

    def __init__(self, service_name: str, config: Optional[RateLimitConfig] = None):
        self.service_name = service_name
        self.limiter = rate_limit_manager.get_limiter(service_name, config)
        logger.info(f"Service wrapper initialized for {service_name}")

    def get_status(self) -> Dict[str, Any]:
        """Get service status including rate limiting info"""
        return self.limiter.get_status()


class DeepgramWrapper(ServiceWrapper):
    """Rate-limited wrapper for Deepgram API"""

    def __init__(self):
        # Use environment-specific configuration
        config = get_service_config("deepgram")
        super().__init__("deepgram", config)
        logger.info(
            f"Deepgram wrapper initialized for {env_config.environment.value} environment"
        )

    @rate_limited("deepgram", Priority.HIGH, Decimal("0.003"))
    def transcribe_audio(
        self,
        audio_path: str,
        job_id: str = "default",
        chunk_duration: Optional[float] = None,
        **options,
    ) -> List[Dict[str, Any]]:
        """Rate-limited audio transcription"""

        # Import here to avoid circular dependencies
        try:
            from ..core.analyze_video import transcribe_deepgram
        except ImportError:
            from analyze_video import transcribe_deepgram

        logger.info(f"Rate-limited Deepgram transcription starting for job {job_id}")

        # Calculate dynamic cost based on audio duration if available
        try:
            from ..utils.ffmpeg_utils import get_video_info

            info = get_video_info(audio_path)
            duration = info.get("duration", 60)  # Default to 60s

            # Deepgram pricing: ~$0.0043 per 15 seconds
            estimated_cost = Decimal(str(duration / 15 * 0.0043))

            # Update cost tracking
            self.limiter.config.cost_per_request = estimated_cost
            logger.debug(
                f"Dynamic cost estimate for {duration}s audio: ${estimated_cost}"
            )

        except Exception as e:
            logger.warning(f"Could not estimate transcription cost: {e}")

        # Execute with error handling
        try:
            result = transcribe_deepgram(audio_path, job_id)
            logger.info(f"Deepgram transcription completed: {len(result)} words")
            return result

        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            raise

    @rate_limited("deepgram_chunked", Priority.NORMAL, Decimal("0.01"))
    def transcribe_large_audio(
        self,
        audio_path: str,
        job_id: str = "default",
        chunk_duration: float = 300.0,  # 5 minutes
        **options,
    ) -> List[Dict[str, Any]]:
        """Rate-limited chunked transcription for large files"""

        try:
            from ..core.analyze_video import transcribe_deepgram_chunked
            from ..utils.ffmpeg_utils import get_video_info
        except ImportError:
            from analyze_video import transcribe_deepgram_chunked
            from utils.ffmpeg_utils import get_video_info

        # Get audio duration for cost estimation
        info = get_video_info(audio_path)
        duration = info.get("duration", 0)

        if duration == 0:
            logger.warning("Could not determine audio duration")
            return []

        # Estimate number of chunks and cost
        estimated_chunks = max(1, int(duration / chunk_duration) + 1)
        estimated_cost = Decimal(str(estimated_chunks * 0.003))

        logger.info(
            f"Starting chunked transcription: {duration}s audio, "
            f"{estimated_chunks} chunks, ~${estimated_cost} estimated cost"
        )

        # Execute chunked transcription
        try:
            result = transcribe_deepgram_chunked(audio_path, job_id, duration)
            logger.info(f"Chunked transcription completed: {len(result)} total words")
            return result

        except Exception as e:
            logger.error(f"Chunked transcription failed: {e}")
            raise


class OpenAIWrapper(ServiceWrapper):
    """Rate-limited wrapper for OpenAI API"""

    def __init__(self):
        # Use environment-specific configuration
        config = get_service_config("openai")
        super().__init__("openai", config)
        logger.info(
            f"OpenAI wrapper initialized for {env_config.environment.value} environment"
        )

    @rate_limited("openai", Priority.HIGH, Decimal("0.02"))
    def generate_completion(
        self, prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 500, **kwargs
    ) -> str:
        """Rate-limited OpenAI completion"""

        import openai
        from ..config import OPENAI_API_KEY

        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")

        # Estimate cost based on tokens (rough approximation)
        estimated_tokens = len(prompt.split()) + max_tokens
        if "gpt-4" in model:
            estimated_cost = Decimal(
                str(estimated_tokens / 1000 * 0.03)
            )  # $0.03/1K tokens
        else:
            estimated_cost = Decimal(
                str(estimated_tokens / 1000 * 0.002)
            )  # $0.002/1K tokens

        logger.info(
            f"OpenAI completion request: {model}, ~{estimated_tokens} tokens, ~${estimated_cost}"
        )

        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                **kwargs,
            )

            result = response.choices[0].message.content
            logger.info(f"OpenAI completion successful: {len(result)} chars returned")
            return result

        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise


class AnthropicWrapper(ServiceWrapper):
    """Rate-limited wrapper for Anthropic Claude API"""

    def __init__(self):
        # Use environment-specific configuration
        config = get_service_config("anthropic")
        super().__init__("anthropic", config)
        logger.info(
            f"Anthropic wrapper initialized for {env_config.environment.value} environment"
        )

    @rate_limited("anthropic", Priority.HIGH, Decimal("0.05"))
    def generate_highlights(
        self, transcript_data: Dict[str, Any], job_id: str = "default", **kwargs
    ) -> Dict[str, Any]:
        """Rate-limited Claude highlight generation"""

        try:
            import anthropic
            from ..config import ANTHROPIC_API_KEY
        except ImportError:
            logger.error("Anthropic client not available")
            raise

        if not ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not configured")

        # Estimate cost based on input size
        input_size = len(str(transcript_data))
        estimated_cost = Decimal(str(input_size / 1000 * 0.01))  # Rough estimate

        logger.info(
            f"Claude highlight generation: {input_size} chars input, ~${estimated_cost}"
        )

        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

            # Prepare prompt for highlight selection
            prompt = self._prepare_highlight_prompt(transcript_data)

            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Cost-effective model
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = response.content[0].text

            # Try to parse JSON response
            import json

            result = json.loads(result_text)

            logger.info(
                f"Claude highlight generation successful: {len(result.get('highlights', []))} highlights"
            )
            return result

        except Exception as e:
            logger.error(f"Claude highlight generation failed: {e}")
            raise

    def _prepare_highlight_prompt(self, transcript_data: Dict[str, Any]) -> str:
        """Prepare prompt for Claude highlight selection"""

        # Extract key information
        transcript = transcript_data.get("transcript", "")
        words = transcript_data.get("words", [])

        # Create simplified transcript with timestamps
        timestamped_segments = []
        if words:
            current_segment = []
            current_start = 0

            for word in words[:100]:  # Limit to first 100 words to save costs
                if not current_segment:
                    current_start = word.get("start", 0)
                current_segment.append(word.get("word", ""))

                # Create segments every 10-15 words
                if len(current_segment) >= 10:
                    timestamped_segments.append(
                        {
                            "start_ms": int(current_start * 1000),
                            "end_ms": int(word.get("end", current_start) * 1000),
                            "text": " ".join(current_segment),
                        }
                    )
                    current_segment = []

        # Format as required by CLAUDE.md specification
        segment_text = "\n".join(
            [
                f"{seg['start_ms']}-{seg['end_ms']}ms: {seg['text']}"
                for seg in timestamped_segments[:20]  # Limit further
            ]
        )

        return f"""Analyze this video transcript and select 3-5 highlight segments.

Transcript segments:
{segment_text}

Return JSON only with this exact structure:
{{
  "highlights": [
    {{
      "slug": "kebab-case-identifier",
      "title": "Short descriptive title (max 8 words)",
      "start_ms": 1000,
      "end_ms": 15000
    }}
  ]
}}

Select segments with:
- Complete sentences or thoughts
- Interesting or important content
- Good audio quality
- 10-25 second duration
- No overlapping times

Respond with JSON only, no other text."""


class GeminiWrapper(ServiceWrapper):
    """Rate-limited wrapper for Google Gemini API"""

    def __init__(self):
        # Use environment-specific configuration
        config = get_service_config("gemini")
        super().__init__("gemini", config)
        logger.info(
            f"Gemini wrapper initialized for {env_config.environment.value} environment"
        )

    @rate_limited("gemini", Priority.NORMAL, Decimal("0.0001"))
    def analyze_story_structure(
        self, transcript_segments: List[Dict], job_id: str = "default", **kwargs
    ) -> Dict[str, Any]:
        """Rate-limited Gemini story analysis"""

        # Import existing function with rate limiting wrapper
        try:
            from ..core.highlight_selector import analyze_story_structure
        except ImportError:
            from highlight_selector import analyze_story_structure

        logger.info(f"Rate-limited Gemini story analysis for job {job_id}")

        try:
            result = analyze_story_structure(transcript_segments, job_id)
            logger.info(
                f"Gemini story analysis completed: {len(result.get('story_beats', []))} beats"
            )
            return result

        except Exception as e:
            logger.error(f"Gemini story analysis failed: {e}")
            raise


# Global service instances
deepgram_service = DeepgramWrapper()
openai_service = OpenAIWrapper()
anthropic_service = AnthropicWrapper()
gemini_service = GeminiWrapper()


def get_service_wrapper(service_name: str) -> ServiceWrapper:
    """Get service wrapper by name"""
    services = {
        "deepgram": deepgram_service,
        "openai": openai_service,
        "anthropic": anthropic_service,
        "gemini": gemini_service,
    }

    if service_name not in services:
        raise ValueError(f"Unknown service: {service_name}")

    return services[service_name]


def get_all_service_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all API services"""
    return {
        "deepgram": deepgram_service.get_status(),
        "openai": openai_service.get_status(),
        "anthropic": anthropic_service.get_status(),
        "gemini": gemini_service.get_status(),
    }


# Convenience functions for backward compatibility
def rate_limited_deepgram_transcribe(*args, **kwargs):
    """Backward compatible wrapper for Deepgram transcription"""
    return deepgram_service.transcribe_audio(*args, **kwargs)


def rate_limited_gemini_analysis(*args, **kwargs):
    """Backward compatible wrapper for Gemini analysis"""
    return gemini_service.analyze_story_structure(*args, **kwargs)


def rate_limited_claude_highlights(*args, **kwargs):
    """Backward compatible wrapper for Claude highlights"""
    return anthropic_service.generate_highlights(*args, **kwargs)


# Graceful degradation handlers
class APIServiceError(Exception):
    """Base exception for API service errors"""

    pass


class RateLimitExceededError(APIServiceError):
    """Raised when rate limits are exceeded"""

    def __init__(self, service_name: str, message: str):
        self.service_name = service_name
        super().__init__(f"Rate limit exceeded for {service_name}: {message}")


class CircuitBreakerOpenError(APIServiceError):
    """Raised when circuit breaker is open"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        super().__init__(
            f"Circuit breaker is open for {service_name} - service unavailable"
        )


def graceful_api_call(
    service_func: Callable,
    fallback_func: Optional[Callable] = None,
    service_name: str = "unknown",
    **kwargs,
) -> Any:
    """Execute API call with graceful degradation"""

    try:
        return service_func(**kwargs)

    except (RateLimitExceededError, CircuitBreakerOpenError, BudgetExceededError) as e:
        logger.warning(f"API service unavailable ({service_name}): {e}")

        if fallback_func:
            logger.info(f"Using fallback for {service_name}")
            return fallback_func(**kwargs)
        else:
            logger.error(f"No fallback available for {service_name}")
            raise

    except Exception as e:
        logger.error(f"Unexpected error in {service_name}: {e}")

        if fallback_func:
            logger.info(f"Using fallback for {service_name} after error")
            return fallback_func(**kwargs)
        else:
            raise


# Health check functionality
def check_service_health() -> Dict[str, Dict[str, Any]]:
    """Check health of all API services"""

    health_status = {}

    for service_name, wrapper in [
        ("deepgram", deepgram_service),
        ("openai", openai_service),
        ("anthropic", anthropic_service),
        ("gemini", gemini_service),
    ]:
        status = wrapper.get_status()

        # Determine health based on circuit breaker and success rate
        is_healthy = (
            status["circuit_breaker_state"] == "CLOSED"
            and status["success_rate"] > 0.8
            and status["consecutive_failures"] < 3
        )

        health_status[service_name] = {
            "healthy": is_healthy,
            "circuit_state": status["circuit_breaker_state"],
            "success_rate": status["success_rate"],
            "queue_size": status["queue_size"],
            "available_tokens": status["available_tokens"],
        }

    return health_status
