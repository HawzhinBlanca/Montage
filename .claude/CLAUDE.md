Primary Directive: Implement the AI Creative Director
Objective: Your sole directive is to implement the Python application known as the "AI Creative Director." This system's purpose is to ingest a long-form video, perform a multi-modal analysis, and produce a professional, actionable Edit Decision List (EDL) in JSON format and a corresponding SRT caption file.

This directive is your constitution. You will not deviate from it.

Core Principles of Engagement (Non-Negotiable)
Phased Execution: You will implement this project by following a strict, sequential order of sub-directives, detailed in the Execution Order section below. You will not proceed to the next directive file until all success criteria for the current one are met.

Use the Right Tool for the Job: You will adhere strictly to the approved technology stack. There will be no substitutions.

No Premature Optimization: Write clean, readable, and correct code first. Performance optimization is not a goal of this phase unless explicitly stated.

Deterministic Logic: Any non-LLM logic, especially scoring and ranking, must be deterministic. Given the same inputs, it must always produce the same output.

Fail Fast and Loud: All functions that interact with external systems (filesystem, APIs, database) must have robust error handling. If a critical step fails, the function must raise a specific exception. Do not silently continue.

Approved Technology Stack (IMPERATIVE)
Language: Python 3.11+

Database: PostgreSQL (via psycopg2-binary)

Video/Audio Analysis: PyAV (av)

AI Models: OpenAI API (via openai)

External Command-Line Tools: ffprobe (via subprocess). You will not use ffmpeg for editing in this phase.

Forbidden Libraries: You are explicitly forbidden from using sqlite3, moviepy, opencv-python (unless for a specific, approved algorithm), or any other video editing library not listed above.

Project Structure
You will work within the existing project structure. You will create new files at the root level as required by the sprint directives. The final structure will look like this:

/Montage
|-- Tasks.md
|-- config.py         # Existing: For configuration
|-- db.py             # Existing: For DB connection pool
|-- docs/             # Existing
|-- migrate.py        # Existing
|-- migrations/       # Existing
|-- requirements.txt  # Existing
|-- tests/            # Existing
|
|-- analysis.py       # TO BE CREATED: All PyAV and AI analysis functions
|-- story_finder.py   # TO BE CREATED: Story identification and ranking logic
|-- edl.py            # TO BE CREATED: EDL and SRT generation
|-- main.py           # TO BE CREATED: Main orchestration script
|-- .env              # TO BE CREATED: For API keys and database URLs

Execution Order
You will now proceed to execute the sub-directives in the following, strict order. A human operator will provide you with these files sequentially. Begin with sprint1_foundation.CLAUDE.md.

sprint1_foundation.CLAUDE.md

sprint2_story_finder.CLAUDE.md

sprint3_edit_plan.CLAUDE.md