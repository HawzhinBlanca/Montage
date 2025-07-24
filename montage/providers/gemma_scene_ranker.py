import json
import os

try:
    from ollama import Client
    _client = Client(os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    OLLAMA_AVAILABLE = True
    print("🚀 Ollama client initialized for local AI processing")
except ImportError:
    # Fallback when ollama is not installed
    OLLAMA_AVAILABLE = False
    _client = None

_SYSTEM = """Rank video segments by importance. Return JSON only: [{'id':0,'importance':0.9}, ...] sorted by importance."""

def rank_scenes_gemma_original(scenes):
    if not OLLAMA_AVAILABLE or _client is None:
        # Fallback: return scenes sorted by existing score
        return [{"id": i, "importance": s.get("score", 0) / 10.0}
                for i, s in enumerate(sorted(scenes, key=lambda x: x.get("score", 0), reverse=True))]

    # Prepare minimal payload for faster processing
    scene_summaries = []
    for i, scene in enumerate(scenes[:10]):  # Limit to 10 scenes max
        summary = {
            "id": i,
            "text": scene.get("text", "")[:100],  # Only 100 chars
            "score": scene.get("score", 0)
        }
        scene_summaries.append(summary)
    
    # Ultra-concise format
    payload = "Rank: " + str(scene_summaries)
    
    for attempt in range(1):  # Single attempt for speed
        try:
            resp = _client.chat(
                model="gemma3:latest",  # Use your installed Gemma3 model
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": payload}
                ],
                options={
                    "temperature": 0.0,  # Deterministic for speed
                    "num_predict": 100,  # Minimal output for <2s response
                    "num_ctx": 512,      # Smaller context for faster processing
                    "num_thread": 8,     # Use multiple threads
                    "f16_kv": True,      # Use f16 for faster inference
                    "use_mlock": True,   # Lock model in memory
                }
            )
            
            # Extract and parse the response
            content = resp["message"]["content"]
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Gemma3 JSON parse error (attempt {attempt + 1}): {e}")
            if attempt == 0:  # Only on first attempt, try to clean the response
                continue
        except Exception as e:
            print(f"⚠️  Gemma3 error (attempt {attempt + 1}): {e}")
            if attempt == 0:
                continue
    
    # Fallback if Gemma3 fails
    print("🔄 Falling back to local scoring after Gemma3 issues")
    return [{"id": i, "importance": s.get("score", 0) / 10.0, "reason": "fallback"}
            for i, s in enumerate(sorted(scenes, key=lambda x: x.get("score", 0), reverse=True))]

# Import fast ranker as primary implementation
from .fast_scene_ranker import rank_scenes
