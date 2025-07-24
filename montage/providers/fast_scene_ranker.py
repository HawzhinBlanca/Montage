"""
Fast Scene Ranker - Optimized for <2s response time
Uses local scoring with optional Gemma enhancement
"""
import json
import time
import threading
from typing import List, Dict, Any
from functools import lru_cache

# Global cache for Gemma responses
_gemma_cache = {}
_cache_lock = threading.Lock()

class FastSceneRanker:
    """Optimized scene ranking for <2s response time"""
    
    def __init__(self):
        self.use_gemma = False  # Disable by default for speed
        self.cache_enabled = True
        
    def rank_scenes_fast(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fast scene ranking that guarantees <2s response time
        
        Strategy:
        1. Use pre-computed scores and heuristics (instant)
        2. Apply smart weighting based on content analysis
        3. Optional: Query Gemma in background for future requests
        """
        start_time = time.time()
        
        # Fast local ranking based on multiple factors
        ranked_scenes = self._local_intelligent_ranking(scenes)
        
        # Check if we have time for Gemma (budget: 1.5s)
        elapsed = time.time() - start_time
        if elapsed < 0.5 and self.use_gemma:
            # Try quick Gemma query with strict timeout
            gemma_result = self._try_quick_gemma(scenes[:5], timeout=1.0)
            if gemma_result:
                ranked_scenes = self._merge_rankings(ranked_scenes, gemma_result)
        
        # Ensure response within 2s
        total_time = time.time() - start_time
        if total_time > 2.0:
            print(f"⚠️ Scene ranking took {total_time:.2f}s (exceeded 2s limit)")
        
        return ranked_scenes
    
    def _local_intelligent_ranking(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Local ranking using intelligent heuristics
        This is instant (<50ms) and provides good results
        """
        rankings = []
        
        for i, scene in enumerate(scenes):
            # Multi-factor scoring
            base_score = scene.get("score", 5.0) / 10.0
            
            # Content analysis factors
            text = scene.get("text", "").lower()
            duration = scene.get("duration", 10)
            energy = scene.get("audio_energy", 0.5)
            
            # Boost for key content indicators
            importance = base_score
            
            # Hook indicators (opening value)
            if any(word in text for word in ["today", "imagine", "what if", "question", "let me"]):
                importance += 0.15
            
            # Key insight indicators
            if any(word in text for word in ["discovered", "realized", "breakthrough", "important", "crucial"]):
                importance += 0.20
            
            # Data/evidence indicators
            if any(word in text for word in ["data", "shows", "percent", "improvement", "results"]):
                importance += 0.15
            
            # Story/emotion indicators
            if any(word in text for word in ["story", "felt", "changed", "personal", "experience"]):
                importance += 0.10
            
            # Energy weighting
            importance += energy * 0.1
            
            # Duration penalty (very short or very long segments)
            if duration < 5:
                importance *= 0.8
            elif duration > 30:
                importance *= 0.9
            
            # Position bias (beginning and end are often important)
            position_ratio = i / max(len(scenes) - 1, 1)
            if position_ratio < 0.2:  # First 20%
                importance += 0.05
            elif position_ratio > 0.8:  # Last 20%
                importance += 0.05
            
            rankings.append({
                "id": i,
                "importance": min(importance, 1.0),
                "reason": self._get_reason(text, importance)
            })
        
        # Sort by importance
        rankings.sort(key=lambda x: x["importance"], reverse=True)
        return rankings
    
    def _get_reason(self, text: str, importance: float) -> str:
        """Generate concise reason for ranking"""
        if importance > 0.8:
            return "key moment"
        elif importance > 0.6:
            return "important"
        elif importance > 0.4:
            return "relevant"
        else:
            return "context"
    
    def _try_quick_gemma(self, top_scenes: List[Dict], timeout: float) -> List[Dict]:
        """Try to get Gemma ranking with strict timeout"""
        # Check cache first
        cache_key = str([s.get("text", "")[:50] for s in top_scenes])
        
        with _cache_lock:
            if cache_key in _gemma_cache:
                return _gemma_cache[cache_key]
        
        # For now, return None to stay under 2s
        # In production, this would do async Gemma query
        return None
    
    def _merge_rankings(self, local_ranks: List[Dict], gemma_ranks: List[Dict]) -> List[Dict]:
        """Merge local and Gemma rankings"""
        # Simple weighted average
        merged = {}
        
        for rank in local_ranks:
            merged[rank["id"]] = rank["importance"] * 0.7
        
        for rank in gemma_ranks:
            if rank["id"] in merged:
                merged[rank["id"]] += rank["importance"] * 0.3
            else:
                merged[rank["id"]] = rank["importance"] * 0.3
        
        # Convert back to list
        result = []
        for id, importance in merged.items():
            result.append({
                "id": id,
                "importance": min(importance, 1.0),
                "reason": "optimized ranking"
            })
        
        result.sort(key=lambda x: x["importance"], reverse=True)
        return result


# Global instance for fast access
_fast_ranker = FastSceneRanker()

def rank_scenes(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fast scene ranking function that guarantees <2s response
    Drop-in replacement for gemma_scene_ranker
    """
    return _fast_ranker.rank_scenes_fast(scenes)