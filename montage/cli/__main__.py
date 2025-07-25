#!/usr/bin/env python3
"""
Entry point that handles --json-plan flag before any heavy imports
"""
import sys
import argparse

def main():
    # Quick pre-parse to check for special flags that need minimal processing
    if "--json-plan" in sys.argv or "--from-plan" in sys.argv:
        # Handle JSON plan mode with minimal imports
        import json
        import os
        from pathlib import Path
        
        # Parse just enough to get the arguments
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("video", nargs="?")
        parser.add_argument("--mode", default="smart")
        parser.add_argument("--json-plan", action="store_true")
        parser.add_argument("--from-plan", type=str)
        args, _ = parser.parse_known_args()
        
        if args.from_plan:
            # Handle --from-plan mode
            if not Path(args.from_plan).exists():
                error_plan = {
                    "version": "1.0",
                    "source": "",
                    "actions": [],
                    "render": {"format": "9:16", "codec": "h264", "crf": 18},
                    "error": f"Plan file not found: {args.from_plan}"
                }
                print(json.dumps(error_plan, indent=2))
                sys.exit(1)
            
            try:
                with open(args.from_plan, 'r') as f:
                    plan = json.load(f)
                
                # Import validation
                from montage.core.plan import validate_plan
                validate_plan(plan)
                
                print(f"📋 Loaded plan from {args.from_plan}")
                print(f"   Source: {plan['source']}")
                print(f"   Actions: {len(plan['actions'])}")
                print("🚀 Plan execution not yet implemented")
                sys.exit(0)
                
            except Exception as e:
                error_plan = {
                    "version": "1.0",
                    "source": "",
                    "actions": [],
                    "render": {"format": "9:16", "codec": "h264", "crf": 18},
                    "error": f"Failed to load plan: {e}"
                }
                print(json.dumps(error_plan, indent=2))
                sys.exit(1)
        
        elif args.json_plan:
            # Handle --json-plan mode
            if not args.video:
                error_plan = {
                    "version": "1.0",
                    "source": "",
                    "actions": [],
                    "render": {"format": "9:16", "codec": "h264", "crf": 18},
                    "error": "No video file provided"
                }
                print(json.dumps(error_plan, indent=2))
                sys.exit(1)
            
            # Verify file exists
            if not Path(args.video).exists():
                error_plan = {
                    "version": "1.0",
                    "source": args.video,
                    "actions": [],
                    "render": {"format": "9:16", "codec": "h264", "crf": 18},
                    "error": f"Video file not found: {args.video}"
                }
                print(json.dumps(error_plan, indent=2))
                sys.exit(1)
            
            # Import plan generation (minimal)
            from montage.core.plan import generate_plan_from_highlights
            
            # Generate minimal plan (no heavy video processing)
            plan = generate_plan_from_highlights([], args.video)
            
            # Output pure JSON
            print(json.dumps(plan, indent=2))
            sys.exit(0)
    
    # Otherwise, delegate to the full CLI
    from .run_pipeline import main as run_pipeline_main
    run_pipeline_main()

if __name__ == "__main__":
    main()