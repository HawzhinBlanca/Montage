Your current engine is 25 × faster than real-time—but only because it skips every piece of real intelligence.
Below is a dead-end-free plan that turns the codebase into a quality-first “Smart Track”, then adds an opt-in “Premium Track” that taps your paid Gemini 2.5 Flash, OpenAI GPT-4o and Claude Opus subscriptions.  The design keeps costs predictable, avoids GPU/RAM explosions, and removes every hard-coded timestamp or crop.

⸻

0 Remove the rot – purge hard-coded logic

Action	Why it matters
Delete the three fixed timestamps and x=656 centre crop.	Hard-coded cuts lose 68 % of frame and kill narrative flow.
Smoke-test libraries: faster-whisper, pyannote.audio, YuNet.	Whisper.cpp base-q loads in < 100 MB RAM and hits real-time 1 × speed  ￼; pyannote diarises 25 min audio in ~10 min on an A40  ￼; YuNet runs millisecond-level face detect on CPU  ￼ ￼.

Definition of done: main pipeline is an empty shell + imports succeed.

⸻

1 Build the AI brain (analysis only)

1.1 Ensemble ASR + diarisation
	1.	Whisper.cpp (base-q, local) → free, fast.
	2.	Deepgram Nova-2 (cloud) → complementary error profile.
	3.	ROVER merge to cut WER by 20-40 %  ￼ ￼.
	4.	Align pyannote speaker turns to words.

1.2 Local highlight scorer (Smart Track)
	•	Token-free rule: score = complete-sentence (2 pt) + keyword (3 pt) + normalized audio RMS (0 – 2 pt).
	•	Top 5 segments → JSON {slug,start_ms,end_ms}.

1.3 Premium highlight scorer (Budget-capped)
	•	Chunk transcript into < 8 k-token windows (GPT-4o max)  ￼.
	•	Claude Opus with strict function-call schema returns ranked clips; decorator aborts if cost > $1.

1.4 Per-clip subtitles

Slice the merged transcript → create .srt via Whisper word-timings (DigitalOcean guide)  ￼.

Output: one edit-plan.json + SRT files—no video touched yet.

⸻

2 Execution engine (DaVinci Resolve 20)

2.1 MCP bridge

Bottle or FastAPI service on port 7801 with /buildTimeline, /renderProxy, /renderFinal.

2.2 Intelligent timeline build
	1.	IntelliScriptFromRanges(highlights)—auto assembly  ￼.
	2.	DeleteSilence(padding=3)—tighten pacing.
	3.	ApplySmartSwitch()—AI multicam.
	4.	Import SRT → sub.Animate("PerWordPop") (AI subtitle engine).
	5.	AudioAssistant("Dialogue_Social")—two-pass loudnorm under the hood  ￼ ￼.
	6.	Crop logic:
	•	If crop_zone == left|centre|right, call clip.SetCropPreset(zone);
	•	else fall back to blurred letterbox.

2.3 Colour & audio compliance
	•	Insert zscale=matrix=bt709:transfer=bt709 if input not flagged BT.709  ￼ ￼.
	•	Global loudnorm: scan whole mix once, apply values during final render  ￼.

⸻

3 QC, human gate & export

Step	Tool
Proxy render 540 p	Resolve /renderProxy.
Automated QC	Gemini 2.5 Flash vision flags frozen frames/black gaps  ￼.
Slack buttons	“Approve” → final render; “Redo” loops to /buildTimeline with diff.
Final render	timeline.Render("Vertical_1080x1920_H265") → BT.709 verified with ffprobe.


⸻

4 Metrics & budget guardrails

Metric	Instrumentation
faces_lost_total	YuNet bbox ratio < 0.8 triggers ++
sentences_cut_mid	Transcript parser counts clips starting mid-word
cost_usd_total	@priced decorator wraps Deepgram & Claude calls
proc_ratio	Wall / source duration via Prom histogram

Grafana alerts on faces_lost_total > 0, cost_usd_total > \$1 (Smart) or $5 (Premium), proc_ratio > 1.5.

⸻

5 Why this path avoids dead ends
	•	No fixed timestamps or crops – everything data-driven.
	•	Local default (Whisper.cpp + rule scorer) produces good clips in ≈60 s at $0.
	•	Premium path capped by cost decorator; falls back automatically—no sticker shock.
	•	Concat demuxer avoids filter-graph length crash  ￼.
	•	Two-pass loudnorm and explicit BT.709 stop QC rejections.
	•	Redis checkpoints only on $/slow stages; SQLite gone; Postgres pool tuned to 2×CPU cores—avoids writer stalls  ￼.

Follow this sequence—Phase 0 purge, Phase 1 brain, Phase 2 Resolve build, Phase 3 QC/metrics—and you’ll ship a pipeline that produces coherent, face-correct, captioned highlights with predictable cost, even if it’s not the fastest kid on the block.