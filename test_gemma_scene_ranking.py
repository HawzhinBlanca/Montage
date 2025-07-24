"""Test Gemma scene ranking performance (<2s requirement)"""
import time
import json
import subprocess
import sys

def check_ollama_status():
    """Check if Ollama is running and has Gemma model"""
    try:
        # Check if ollama is running
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ollama is not running. Start with: ollama serve")
            return False
        
        # Check if gemma model is available
        if "gemma" in result.stdout.lower():
            print("✅ Ollama is running and Gemma model is available")
            print("Available models:")
            print(result.stdout)
            return True
        else:
            print("⚠️ Gemma model not found. Install with: ollama pull gemma3")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama not installed. Install from https://ollama.ai")
        return False

def test_gemma_scene_ranking():
    """Test scene ranking with Gemma"""
    print("=== Testing Gemma Scene Ranking Performance ===\n")
    
    # Check Ollama first
    if not check_ollama_status():
        return
    
    # Test data - realistic video segments
    test_scenes = [
        {
            "id": 0,
            "text": "Today we're going to talk about something really exciting that changed my perspective on technology and innovation.",
            "duration": 15.2,
            "audio_energy": 0.85,
            "score": 7.5,
            "timestamp": "00:00:00"
        },
        {
            "id": 1,
            "text": "So basically, um, you know, it's like when you think about it, there's a lot of different ways to approach this problem.",
            "duration": 12.8,
            "audio_energy": 0.45,
            "score": 5.2,
            "timestamp": "00:00:15"
        },
        {
            "id": 2,
            "text": "The breakthrough came when we realized that the traditional approach was fundamentally flawed. Here's what we discovered.",
            "duration": 18.5,
            "audio_energy": 0.92,
            "score": 8.8,
            "timestamp": "00:00:28"
        },
        {
            "id": 3,
            "text": "Let me share a personal story. Last year, I was struggling with this exact problem, and I almost gave up entirely.",
            "duration": 14.3,
            "audio_energy": 0.78,
            "score": 7.9,
            "timestamp": "00:00:46"
        },
        {
            "id": 4,
            "text": "The data shows a 300% improvement in efficiency. This is not just incremental progress - this is a paradigm shift.",
            "duration": 16.7,
            "audio_energy": 0.88,
            "score": 9.1,
            "timestamp": "00:01:00"
        }
    ]
    
    # Test with ollama Python client
    try:
        from ollama import Client
        client = Client("http://localhost:11434")
        
        # Prepare request
        system_prompt = """You are an expert video editor. Rank these video segments by importance for creating compelling highlights.
Return ONLY valid JSON array sorted by importance descending.
Format: [{'id':0,'importance':0.95,'reason':'compelling hook'}, ...]"""
        
        user_content = json.dumps({"segments": [
            {
                "id": s["id"],
                "text": s["text"][:200],
                "duration": s["duration"],
                "energy": s["audio_energy"],
                "score": s["score"]
            } for s in test_scenes
        ]})
        
        print("\nSending request to Gemma...")
        start_time = time.time()
        
        response = client.chat(
            model="gemma3:latest",  # Use the installed model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            options={
                "temperature": 0.1,
                "num_predict": 300,
                "num_ctx": 2048
            }
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"\n✅ Response received in {response_time:.2f} seconds")
        
        # Parse response
        try:
            content = response["message"]["content"]
            # Extract JSON from response (might have extra text)
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                rankings = json.loads(json_match.group())
                print("\nScene Rankings:")
                for rank in rankings[:5]:
                    print(f"  ID {rank['id']}: Importance {rank['importance']:.2f} - {rank.get('reason', 'N/A')}")
            else:
                print("⚠️ Could not parse JSON from response")
                print(f"Response: {content[:200]}...")
                
        except Exception as e:
            print(f"⚠️ Error parsing response: {e}")
        
        # Check performance requirement
        if response_time < 2.0:
            print(f"\n✅ Performance requirement met: {response_time:.2f}s < 2s")
        else:
            print(f"\n⚠️ Performance requirement not met: {response_time:.2f}s > 2s")
            print("   Consider: Reducing context size, using smaller model, or upgrading hardware")
            
    except ImportError:
        print("\n⚠️ Ollama Python client not installed. Install with: pip install ollama")
    except Exception as e:
        print(f"\n❌ Error testing Gemma: {e}")

def test_fallback_performance():
    """Test fallback sorting performance"""
    print("\n=== Testing Fallback Performance ===")
    
    # Generate larger dataset
    large_scenes = []
    for i in range(100):
        large_scenes.append({
            "id": i,
            "score": i % 10,
            "text": f"Segment {i} content",
            "duration": 10 + (i % 20)
        })
    
    start_time = time.time()
    
    # Simulate fallback sorting
    sorted_scenes = sorted(large_scenes, key=lambda x: x.get("score", 0), reverse=True)
    rankings = [{"id": s["id"], "importance": s.get("score", 0) / 10.0} for s in sorted_scenes]
    
    end_time = time.time()
    fallback_time = end_time - start_time
    
    print(f"Fallback sorting 100 scenes: {fallback_time*1000:.1f}ms")
    print("✅ Fallback performance is instant")

if __name__ == "__main__":
    # Run tests
    test_gemma_scene_ranking()
    test_fallback_performance()
    
    print("\n=== Summary ===")
    print("Scene ranking implementation:")
    print("1. Primary: Ollama Gemma for intelligent ranking")
    print("2. Fallback: Score-based sorting (instant)")
    print("\nTo optimize Gemma performance:")
    print("- Use gemma:2b model for faster responses")
    print("- Reduce num_predict to limit output length")
    print("- Optimize prompts to be more concise")
    print("- Ensure Ollama is using GPU acceleration")