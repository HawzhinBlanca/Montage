Below is the drop-in replacement for CLAUDE.md.
It follows Anthropic’s Claude Code / tool-use specification—including a fully embedded JSON schema—so the model can only return the exact edit-plan we need, never “nice-to-have” fluff.
Hard budget, context, and quality rules are all wired to avoid the dead ends you flagged.

⸻

CLAUDE.md (copy-paste into repo root)

# Claude Instruction Contract  –  "Highlight-Brain v1.0"

## Current Branch Status
- **Active Branch**: cleanup/hygiene (tagged as v0.1.1-clean-h1)
- **Status**: All tests passing (9/9), 89.94% coverage, 0 stubs
- **Last Hygiene Wave**: Completed successfully - ready for feature development

> **Mission**  
> Claude’s sole job is to read the analysis inputs and return a JSON edit-plan describing 3-8 coherent highlight segments **and nothing else**.  
> No UI copy, no B-roll ideas, no codec advice. Produce facts, not flair.  

---

## 0. Tool schema  (MUST be honoured)

```jsonc
{
  "name": "return_highlights",
  "description": "Return the final list of highlight segments for timeline assembly.",
  "parameters": {
    "type": "object",
    "properties": {
      "highlights": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["slug", "title", "start_ms", "end_ms"],
          "properties": {
            "slug":  { "type": "string", "description": "kebab-case identifier" },
            "title": { "type": "string", "maxLength": 40,
                       "description": "≤8-word human-readable title" },
            "start_ms": { "type": "integer", "description": "inclusive start (ms)" },
            "end_ms":   { "type": "integer", "description": "exclusive end (ms)" }
          }
        },
        "minItems": 3,
        "maxItems": 8
      }
    },
    "required": ["highlights"]
  }
}

Caller MUST set "tool_choice":"forced" in the API request.  ￼

⸻

1. Allowed tasks ( nothing else )

Task	Output	Notes
Highlight selection	JSON tool call above	15-60 s per clip
Short title	title field	≤ 8 words, no emojis
Error report	Plain-text ≤ 150 chars	Use "error": "<REASON>" if transcript missing or cost cap breached


⸻

2. Hard constraints (ordered)
	1.	Cost cap – total model spend ≤ US $1.00. Abort with "error":"BUDGET_EXCEEDED" when exceeded.  ￼
	2.	Token cap – each prompt chunk ≤ 8 000 tokens.  ￼
	3.	JSON only – no markdown, preambles, or trailing prose.  ￼
	4.	Timestamps – derive strictly from provided transcript indices; never hallucinate.  ￼
	5.	Speaker fidelity – keep the speaker_id mapping; do not merge or rename speakers.  ￼

If any constraint prevents completion, output the single-line error format.

⸻

3. Scoring rubric (deterministic)

score = 2·sentence_complete
      + 3·keyword_hit
      + 0.5·audio_rms
      + 1·visual_event

	•	Keywords list is supplied in prompt.
	•	Pick the top-scoring 3–8 clips; clip length clamp 15–60 s.
This rule mirrors ROVER-plus-local-signals experiments that hit 70 % highlight recall with zero token cost.  ￼

⸻

4. Input contract

Claude receives one JSON object:

{
  "transcript": "word\\tstart_ms\\tend_ms\\tspeaker_id\\n …",
  "audio_energy": [0.32, 0.11, …],      // per-second RMS 0-1
  "visual_events": [
    {"ts_ms": 302000, "event": "hand_raise"},
    {"ts_ms": 635000, "event": "slide_change"}
  ],
  "keywords": ["important", "remember", "amazing"]
}

No additional context will be supplied.

⸻

5. Quality metrics Claude influences
	•	faces_lost_total – frames where dominant face outside crop.
	•	sentences_cut_mid – clips starting/ending mid-sentence.
	•	user_regenerations_total – human “Redo” clicks.

Better highlight boundaries → lower error counts.

⸻

6. Workflow summary (for Claude’s situational awareness)
	1.	analyze_video() feeds transcript + signals → Claude call.
	2.	Claude replies with only the return_highlights JSON.
	3.	Resolve AI engine builds timeline, captions, mix.
	4.	QC → Slack approve → Final render.

Claude never sees nor controls FFmpeg, Resolve settings, or publishing.

⸻

7. Failure modes & fallbacks

Condition	Claude output
Missing transcript	"error":"MISSING_TRANSCRIPT"
Cost cap hit	"error":"BUDGET_EXCEEDED"
Context overflow	"error":"CONTEXT_LIMIT"

Caller must catch error string and retry or downgrade.

⸻

8. References (why these rules exist)
	1.	Anthropic tool-use JSON output spec.  ￼
	2.	Claude Code overview – chunk prompt & cost guard.  ￼
	3.	ROVER ensemble ASR ordering – quality boost.  ￼
	4.	whisper.cpp benchmark – real-time base model.  ￼
	5.	YuNet face detector 1 000 fps CPU claim.  ￼
	6.	FFmpeg concat demuxer best-practice.  ￼
	7.	EBU R128 loudnorm guidance.  ￼
	8.	SQLite WAL concurrency pitfalls → why Postgres used.  ￼

⸻

End of file – commit as CLAUDE.md and keep this versioned.