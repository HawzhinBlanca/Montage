#!/usr/bin/env python3
"""
Show current configuration (with secrets masked)
"""
import json
import sys
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from ..settings import get_settings


def show_config(format: str = "tree"):
    """Display current configuration"""
    console = Console()
    settings = get_settings()
    
    if format == "json":
        # JSON output
        config = settings.get_safe_dict()
        console.print_json(data=config)
    
    elif format == "env":
        # Environment variable format
        console.print("[bold blue]Environment Variables:[/bold blue]")
        console.print()
        
        def flatten_dict(d, prefix=""):
            items = []
            for k, v in d.items():
                new_key = f"{prefix}__{k}".upper() if prefix else k.upper()
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key))
                else:
                    items.append((new_key, v))
            return items
        
        for key, value in flatten_dict(settings.get_safe_dict()):
            console.print(f"{key}={value}")
    
    else:  # tree format (default)
        # Tree view
        tree = Tree("[bold blue]Montage Configuration[/bold blue]")
        
        # Application info
        app_tree = tree.add("[cyan]Application[/cyan]")
        app_tree.add(f"Name: {settings.app_name}")
        app_tree.add(f"Version: {settings.app_version}")
        app_tree.add(f"Environment: {settings.environment}")
        app_tree.add(f"Debug: {settings.debug}")
        
        # Database
        db_tree = tree.add("[cyan]Database[/cyan]")
        db_tree.add(f"URL: ***MASKED***")
        db_tree.add(f"Pool Size: {settings.database.pool_size}")
        db_tree.add(f"Max Overflow: {settings.database.max_overflow}")
        
        # Redis
        redis_tree = tree.add("[cyan]Redis[/cyan]")
        redis_tree.add(f"URL: ***MASKED***")
        redis_tree.add(f"Max Connections: {settings.redis.max_connections}")
        
        # API Keys
        api_tree = tree.add("[cyan]API Keys[/cyan]")
        api_tree.add(f"OpenAI: {'✅ Configured' if settings.api_keys.has_openai else '❌ Not Set'}")
        api_tree.add(f"Anthropic: {'✅ Configured' if settings.api_keys.has_anthropic else '❌ Not Set'}")
        api_tree.add(f"Deepgram: {'✅ Configured' if settings.api_keys.has_deepgram else '❌ Not Set'}")
        api_tree.add(f"Gemini: {'✅ Configured' if settings.api_keys.has_gemini else '❌ Not Set'}")
        
        # Cost Limits
        cost_tree = tree.add("[cyan]Cost Limits[/cyan]")
        cost_tree.add(f"Max Cost USD: ${settings.costs.max_cost_usd}")
        cost_tree.add(f"Max Tokens: {settings.costs.max_tokens_per_request}")
        cost_tree.add(f"OpenAI Daily: ${settings.costs.openai_daily_limit}")
        cost_tree.add(f"Anthropic Daily: ${settings.costs.anthropic_daily_limit}")
        
        # Video Processing
        video_tree = tree.add("[cyan]Video Processing[/cyan]")
        video_tree.add(f"Max File Size: {settings.video.max_file_size_mb} MB")
        video_tree.add(f"Max Duration: {settings.video.max_duration_seconds} seconds")
        video_tree.add(f"Formats: {', '.join(settings.video.allowed_formats)}")
        video_tree.add(f"VMAF Enabled: {settings.video.enable_vmaf}")
        video_tree.add(f"Audio Normalization: {settings.video.enable_audio_normalization}")
        
        # Security
        security_tree = tree.add("[cyan]Security[/cyan]")
        security_tree.add(f"Path Sanitization: {settings.security.enable_path_sanitization}")
        security_tree.add(f"SQL Protection: {settings.security.enable_sql_injection_protection}")
        security_tree.add(f"Rate Limiting: {settings.security.rate_limit_enabled}")
        security_tree.add(f"JWT Secret: ***MASKED***")
        
        # Feature Flags
        features_tree = tree.add("[cyan]Feature Flags[/cyan]")
        features_tree.add(f"Speaker Diarization: {settings.features.enable_speaker_diarization}")
        features_tree.add(f"Smart Crop: {settings.features.enable_smart_crop}")
        features_tree.add(f"AB Testing: {settings.features.enable_ab_testing}")
        features_tree.add(f"Caching: {settings.features.enable_caching}")
        
        console.print(tree)
        
        # Warnings
        console.print()
        if settings.environment == "development" and settings.debug:
            console.print("[yellow]⚠️  Debug mode is enabled in development[/yellow]")
        if not settings.api_keys.has_openai and not settings.api_keys.has_anthropic:
            console.print("[red]❌ No AI API keys configured - premium features disabled[/red]")
        if not settings.monitoring.sentry_dsn and settings.environment == "production":
            console.print("[red]❌ Sentry not configured for production environment[/red]")


def main():
    """CLI entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Show current Montage configuration")
    parser.add_argument(
        "--format", 
        choices=["tree", "json", "env"], 
        default="tree",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    try:
        show_config(args.format)
    except Exception as e:
        console = Console()
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()